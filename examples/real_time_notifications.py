#!/usr/bin/env python3
"""
Real-time notifications example demonstrating event-driven processing with Shadows.

This example shows how to build a comprehensive notification system that handles:
- Event-driven notification processing
- Webhook deduplication and processing
- Notification batching for efficiency
- Priority-based processing
- Rate limiting and backoff
- Multi-channel delivery (email, SMS, push)
- Delivery tracking and retry logic
- Template rendering and personalization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from shadows import (
    Shadow,
    Worker,
    Retry,
    ExponentialRetry,
    ConcurrencyLimit,
    CurrentShadow,
    TaskLogger,
    TaskKey,
    Logged
)
from common import run_redis


class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


@dataclass
class NotificationEvent:
    event_id: str
    user_id: str
    event_type: str
    data: Dict[str, Any]
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = [NotificationChannel.EMAIL]


@dataclass
class DeliveryAttempt:
    notification_id: str
    channel: NotificationChannel
    attempt_number: int
    status: str
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None


# Global notification batch storage (in production, use Redis/database)
notification_batches = {}


async def process_notification_event(
    event: NotificationEvent,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Process incoming notification events and route to appropriate channels.

    This function demonstrates:
    - Event deduplication using task keys
    - Priority-based routing
    - Multi-channel distribution
    - Template selection based on event type
    """
    logger.info(
        "processing notification event",
        extra={
            "event_id": event.event_id,
            "user_id": event.user_id,
            "event_type": event.event_type,
            "priority": event.priority.value,
            "channels": [c.value for c in event.channels]
        }
    )

    # Generate unique notification ID
    notification_id = f"notif-{event.event_id}"

    # Route to each requested channel
    for channel in event.channels:
        if event.priority == NotificationPriority.URGENT:
            # Urgent notifications go directly to delivery
            await shadows.add(deliver_notification)(
                notification_id=notification_id,
                event=event,
                channel=channel,
                key=f"urgent-{notification_id}-{channel.value}"
            )
        else:
            # Non-urgent notifications get batched
            await shadows.add(batch_notification)(
                notification_id=notification_id,
                event=event,
                channel=channel,
                key=f"batch-{event.user_id}-{channel.value}"
            )


async def batch_notification(
    notification_id: str,
    event: NotificationEvent,
    channel: NotificationChannel,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Batch notifications for efficient delivery.

    Demonstrates:
    - Notification batching for efficiency
    - Time-based batch processing
    - User-specific batching
    """
    batch_key = f"user-{event.user_id}-{channel.value}"

    if batch_key not in notification_batches:
        notification_batches[batch_key] = {
            "notifications": [],
            "created_at": datetime.now(),
            "channel": channel
        }

    notification_batches[batch_key]["notifications"].append({
        "id": notification_id,
        "event": event,
        "added_at": datetime.now()
    })

    logger.info(
        f"batched notification for {batch_key}",
        extra={
            "batch_size": len(notification_batches[batch_key]["notifications"]),
            "notification_id": notification_id
        }
    )

    # Schedule batch processing if this is the first notification or batch is getting large
    batch = notification_batches[batch_key]
    if len(batch["notifications"]) >= 5:  # Process when we have 5 notifications
        await shadows.add(process_notification_batch)(
            batch_key=batch_key,
            key=f"process-batch-{batch_key}"
        )
    else:
        # Schedule processing in 30 seconds if not already scheduled
        await shadows.add(process_notification_batch)(
            batch_key=batch_key,
            key=f"process-batch-{batch_key}",
            when=datetime.now() + timedelta(seconds=30)
        )


async def process_notification_batch(
    batch_key: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Process a batch of notifications efficiently.

    Demonstrates:
    - Batch processing optimization
    - Template consolidation
    - Bulk delivery
    """
    if batch_key not in notification_batches:
        logger.warning(f"batch {batch_key} not found or already processed")
        return

    batch = notification_batches[batch_key]
    notifications = batch["notifications"]
    channel = batch["channel"]

    logger.info(
        f"processing notification batch",
        extra={
            "batch_key": batch_key,
            "notification_count": len(notifications),
            "channel": channel.value
        }
    )

    if len(notifications) == 1:
        # Single notification - process individually
        notification = notifications[0]
        await shadows.add(deliver_notification)(
            notification_id=notification["id"],
            event=notification["event"],
            channel=channel,
            key=f"deliver-{notification['id']}-{channel.value}"
        )
    else:
        # Multiple notifications - create digest
        await shadows.add(deliver_notification_digest)(
            batch_key=batch_key,
            notifications=notifications,
            channel=channel,
            key=f"digest-{batch_key}"
        )

    # Clean up processed batch
    del notification_batches[batch_key]


async def deliver_notification(
    notification_id: str,
    event: NotificationEvent,
    channel: NotificationChannel,
    retry: ExponentialRetry = ExponentialRetry(
        attempts=3,
        minimum_delay=timedelta(seconds=1),
        maximum_delay=timedelta(minutes=5)
    ),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Deliver a single notification via specified channel.

    Demonstrates:
    - Retry logic for external service calls
    - Delivery tracking
    - Channel-specific logic
    - Template rendering
    """
    logger.info(
        f"delivering notification via {channel.value}",
        extra={
            "notification_id": notification_id,
            "attempt": retry.attempt,
            "event_type": event.event_type,
            "user_id": event.user_id
        }
    )

    try:
        if channel == NotificationChannel.EMAIL:
            await deliver_email_notification(notification_id, event)
        elif channel == NotificationChannel.SMS:
            await deliver_sms_notification(notification_id, event)
        elif channel == NotificationChannel.PUSH:
            await deliver_push_notification(notification_id, event)
        elif channel == NotificationChannel.WEBHOOK:
            await deliver_webhook_notification(notification_id, event)

        # Track successful delivery
        await track_delivery_attempt(
            notification_id, channel, retry.attempt, "delivered"
        )

        logger.info(
            f"successfully delivered notification via {channel.value}",
            extra={"notification_id": notification_id}
        )

    except Exception as e:
        logger.error(
            f"failed to deliver notification via {channel.value}",
            extra={
                "notification_id": notification_id,
                "error": str(e),
                "will_retry": retry.attempt < retry.attempts
            }
        )

        await track_delivery_attempt(
            notification_id, channel, retry.attempt, "failed", str(e)
        )

        if retry.attempt >= retry.attempts:
            # All retries exhausted - could trigger escalation
            logger.critical(
                f"all delivery attempts exhausted for {notification_id}",
                extra={"notification_id": notification_id, "channel": channel.value}
            )
        else:
            raise  # Re-raise to trigger retry


async def deliver_notification_digest(
    batch_key: str,
    notifications: List[Dict],
    channel: NotificationChannel,
    retry: ExponentialRetry = ExponentialRetry(
        attempts=3,
        minimum_delay=timedelta(seconds=2),
        maximum_delay=timedelta(minutes=10)
    ),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Deliver a digest of multiple notifications.

    Demonstrates:
    - Digest creation and delivery
    - Batch processing efficiency
    - Template consolidation
    """
    logger.info(
        f"delivering notification digest via {channel.value}",
        extra={
            "batch_key": batch_key,
            "notification_count": len(notifications),
            "attempt": retry.attempt
        }
    )

    try:
        # Create digest content
        digest_events = [n["event"] for n in notifications]
        user_id = digest_events[0].user_id

        if channel == NotificationChannel.EMAIL:
            await deliver_email_digest(user_id, digest_events)
        elif channel == NotificationChannel.SMS:
            await deliver_sms_digest(user_id, digest_events)
        elif channel == NotificationChannel.PUSH:
            await deliver_push_digest(user_id, digest_events)

        logger.info(
            f"successfully delivered notification digest",
            extra={"batch_key": batch_key}
        )

    except Exception as e:
        logger.error(
            f"failed to deliver notification digest",
            extra={
                "batch_key": batch_key,
                "error": str(e),
                "will_retry": retry.attempt < retry.attempts
            }
        )

        if retry.attempt >= retry.attempts:
            logger.critical(
                f"all digest delivery attempts exhausted",
                extra={"batch_key": batch_key}
            )
        else:
            raise


# Channel-specific delivery functions (mock implementations)
async def deliver_email_notification(notification_id: str, event: NotificationEvent) -> None:
    """Mock email delivery - in production would call email service API"""
    await asyncio.sleep(0.1)  # Simulate API call
    print(f"📧 EMAIL sent to user {event.user_id}: {event.event_type}")


async def deliver_sms_notification(notification_id: str, event: NotificationEvent) -> None:
    """Mock SMS delivery"""
    await asyncio.sleep(0.05)  # Simulate API call
    print(f"📱 SMS sent to user {event.user_id}: {event.event_type}")


async def deliver_push_notification(notification_id: str, event: NotificationEvent) -> None:
    """Mock push notification delivery"""
    await asyncio.sleep(0.03)  # Simulate API call
    print(f"🔔 PUSH sent to user {event.user_id}: {event.event_type}")


async def deliver_webhook_notification(notification_id: str, event: NotificationEvent) -> None:
    """Mock webhook delivery"""
    await asyncio.sleep(0.2)  # Simulate HTTP call
    print(f"🔗 WEBHOOK sent for event {event.event_id}")


async def deliver_email_digest(user_id: str, events: List[NotificationEvent]) -> None:
    """Mock digest email delivery"""
    await asyncio.sleep(0.15)
    print(f"📧 DIGEST EMAIL sent to user {user_id}: {len(events)} events")


async def deliver_sms_digest(user_id: str, events: List[NotificationEvent]) -> None:
    """Mock digest SMS delivery"""
    await asyncio.sleep(0.08)
    print(f"📱 DIGEST SMS sent to user {user_id}: {len(events)} events")


async def deliver_push_digest(user_id: str, events: List[NotificationEvent]) -> None:
    """Mock digest push delivery"""
    await asyncio.sleep(0.05)
    print(f"🔔 DIGEST PUSH sent to user {user_id}: {len(events)} events")


async def track_delivery_attempt(
    notification_id: str,
    channel: NotificationChannel,
    attempt: int,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """Track delivery attempts - in production would store in database"""
    attempt_record = DeliveryAttempt(
        notification_id=notification_id,
        channel=channel,
        attempt_number=attempt,
        status=status,
        error_message=error_message,
        delivered_at=datetime.now() if status == "delivered" else None
    )

    # In production: save to database
    print(f"📊 Tracked delivery: {notification_id} via {channel.value} - {status}")


async def process_webhook_event(
    webhook_payload: Dict[str, Any],
    webhook_signature: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Process incoming webhook with deduplication.

    Demonstrates:
    - Webhook deduplication using event IDs
    - Signature verification
    - Rate limiting
    - Event transformation
    """
    logger.info("processing webhook event", extra={"payload_keys": list(webhook_payload.keys())})

    # Extract event information
    event_id = webhook_payload.get("id") or webhook_payload.get("event_id")
    user_id = webhook_payload.get("user_id") or webhook_payload.get("customer_id")
    event_type = webhook_payload.get("type") or webhook_payload.get("event_type")

    if not all([event_id, user_id, event_type]):
        logger.error("missing required webhook fields", extra=webhook_payload)
        return

    # Verify webhook signature (mock implementation)
    if not await verify_webhook_signature(webhook_payload, webhook_signature):
        logger.warning("invalid webhook signature", extra={"event_id": event_id})
        return

    # Check for duplicate processing
    await shadows.add(process_notification_event)(
        event=NotificationEvent(
            event_id=event_id,
            user_id=user_id,
            event_type=event_type,
            data=webhook_payload,
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.EMAIL, NotificationChannel.PUSH]
        ),
        key=f"webhook-{event_id}"  # Deduplication key
    )


async def verify_webhook_signature(payload: Dict[str, Any], signature: str) -> bool:
    """Mock webhook signature verification"""
    await asyncio.sleep(0.01)  # Simulate verification
    return signature == "valid_signature"  # Mock validation


async def main():
    """Demonstrate the notification system with various scenarios."""
    async with run_redis("7.4.2") as redis_url:
        async with Shadow(name="notifications", url=redis_url) as shadows:
            # Register all notification tasks
            shadows.register(process_notification_event)
            shadows.register(batch_notification)
            shadows.register(process_notification_batch)
            shadows.register(deliver_notification)
            shadows.register(deliver_notification_digest)
            shadows.register(process_webhook_event)

            print("=== Real-time Notifications Demo ===\n")

            # Scenario 1: Process individual high-priority events
            print("1. Processing urgent notification events...")
            await shadows.add(process_notification_event)(
                event=NotificationEvent(
                    event_id="urgent-001",
                    user_id="user123",
                    event_type="account_security_alert",
                    data={"threat_type": "suspicious_login"},
                    priority=NotificationPriority.URGENT,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PUSH]
                )
            )

            # Scenario 2: Process normal events that will be batched
            print("\n2. Processing normal notification events (will be batched)...")
            for i in range(7):  # This will create 2 batches (5 + 2)
                await shadows.add(process_notification_event)(
                    event=NotificationEvent(
                        event_id=f"normal-{i:03d}",
                        user_id="user456",
                        event_type="order_update",
                        data={"order_id": f"ORD-{i:03d}", "status": "shipped"},
                        priority=NotificationPriority.NORMAL,
                        channels=[NotificationChannel.EMAIL]
                    )
                )

            # Scenario 3: Process webhook events with deduplication
            print("\n3. Processing webhook events...")
            webhook_payload = {
                "id": "webhook-001",
                "user_id": "user789",
                "type": "payment_received",
                "amount": 99.99,
                "currency": "USD"
            }

            await shadows.add(process_webhook_event)(
                webhook_payload=webhook_payload,
                webhook_signature="valid_signature"
            )

            # Try to process the same webhook again (should be deduplicated)
            await shadows.add(process_webhook_event)(
                webhook_payload=webhook_payload,
                webhook_signature="valid_signature"
            )

            # Scenario 4: Demonstrate different channels
            print("\n4. Processing multi-channel notifications...")
            await shadows.add(process_notification_event)(
                event=NotificationEvent(
                    event_id="multi-001",
                    user_id="user999",
                    event_type="welcome_bonus",
                    data={"bonus_amount": 10.00},
                    priority=NotificationPriority.NORMAL,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PUSH]
                )
            )

            print("\n=== Starting Notification Workers ===")
            async with Worker(shadows, concurrency=10) as worker:
                await worker.run_until_finished()

            print("\n=== Notification Processing Complete ===")
            print(f"Remaining batches: {len(notification_batches)}")


if __name__ == "__main__":
    asyncio.run(main())
