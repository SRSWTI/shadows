"""Tests for real-time notifications example.

This module tests the notification system's functionality including:
- Event-driven notification processing
- Webhook deduplication and processing
- Notification batching for efficiency
- Multi-channel delivery
- Retry logic and error handling
- Delivery tracking
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from shadows import Shadow, Worker, ExponentialRetry
from shadows.examples.real_time_notifications import (
    NotificationEvent,
    NotificationChannel,
    ProcessingStatus,
    DeliveryAttempt,
    file_registry,
    processing_results,
    notification_batches,
    process_notification_event,
    batch_notification,
    process_notification_batch,
    deliver_notification,
    deliver_notification_digest,
    process_webhook_event,
    deliver_email_notification,
    deliver_sms_notification,
    deliver_push_notification,
    deliver_webhook_notification,
    track_delivery_attempt,
    verify_webhook_signature,
    send_processing_notification,
    cleanup_failed_file
)


@pytest.fixture
def notification_event():
    """Create a test notification event."""
    return NotificationEvent(
        event_id="test-event-123",
        user_id="user456",
        event_type="order_shipped",
        data={"order_id": "ORD-789", "tracking_number": "1Z999AA1234567890"},
        priority=NotificationEvent.NotificationPriority.NORMAL,
        channels=[NotificationChannel.EMAIL, NotificationChannel.PUSH]
    )


@pytest.fixture
def webhook_payload():
    """Create a test webhook payload."""
    return {
        "id": "webhook-123",
        "user_id": "user789",
        "type": "payment_completed",
        "amount": 99.99,
        "currency": "USD",
        "timestamp": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def sample_batch():
    """Create a sample notification batch."""
    return {
        "notifications": [
            {
                "id": "notif-1",
                "event": NotificationEvent(
                    event_id="event-1",
                    user_id="user123",
                    event_type="test_event",
                    data={"key": "value"}
                ),
                "added_at": datetime.now(timezone.utc)
            }
        ],
        "created_at": datetime.now(timezone.utc),
        "channel": NotificationChannel.EMAIL
    }


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state between tests."""
    file_registry.clear()
    processing_results.clear()
    notification_batches.clear()


class TestNotificationEvent:
    """Test NotificationEvent dataclass."""

    def test_notification_event_creation(self):
        """Test creating a notification event."""
        event = NotificationEvent(
            event_id="test-123",
            user_id="user456",
            event_type="order_update",
            data={"order_id": "ORD-789"}
        )

        assert event.event_id == "test-123"
        assert event.user_id == "user456"
        assert event.event_type == "order_update"
        assert event.data["order_id"] == "ORD-789"
        assert event.priority == NotificationEvent.NotificationPriority.NORMAL
        assert event.channels == [NotificationChannel.EMAIL]

    def test_notification_event_with_custom_channels(self):
        """Test notification event with custom channels."""
        event = NotificationEvent(
            event_id="test-123",
            user_id="user456",
            event_type="urgent_alert",
            data={"alert": "system_down"},
            channels=[NotificationChannel.SMS, NotificationChannel.PUSH]
        )

        assert event.channels == [NotificationChannel.SMS, NotificationChannel.PUSH]


class TestProcessNotificationEvent:
    """Test notification event processing."""

    @pytest.mark.asyncio
    async def test_process_notification_event_urgent(self, shadows: Shadow, notification_event):
        """Test processing urgent notification events."""
        notification_event.priority = NotificationEvent.NotificationPriority.URGENT

        # Mock the add method to capture calls
        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            await process_notification_event(notification_event, shadows)

            # Should call deliver_notification directly for urgent notifications
            assert mock_add.call_count == 2  # One for each channel

            # Check calls
            calls = mock_add.call_args_list
            assert "deliver_notification" in str(calls[0])
            assert "deliver_notification" in str(calls[1])

    @pytest.mark.asyncio
    async def test_process_notification_event_normal(self, shadows: Shadow, notification_event):
        """Test processing normal notification events."""
        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            await process_notification_event(notification_event, shadows)

            # Should call batch_notification for normal notifications
            assert mock_add.call_count == 2  # One for each channel

            calls = mock_add.call_args_list
            assert "batch_notification" in str(calls[0])
            assert "batch_notification" in str(calls[1])


class TestBatchNotification:
    """Test notification batching functionality."""

    @pytest.mark.asyncio
    async def test_batch_notification_single(self, shadows: Shadow, notification_event):
        """Test batching a single notification."""
        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            await batch_notification(
                notification_id="notif-123",
                event=notification_event,
                channel=NotificationChannel.EMAIL,
                shadows=shadows
            )

            # Should have one notification in batch
            batch_key = f"user-{notification_event.user_id}-{NotificationChannel.EMAIL.value}"
            assert batch_key in notification_batches
            assert len(notification_batches[batch_key]["notifications"]) == 1

            # Should schedule batch processing
            mock_add.assert_called_once()
            assert "process_notification_batch" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_batch_notification_multiple(self, shadows: Shadow, notification_event):
        """Test batching multiple notifications."""
        batch_key = f"user-{notification_event.user_id}-{NotificationChannel.EMAIL.value}"

        # Add first notification
        with patch.object(shadows, 'add', new_callable=AsyncMock):
            await batch_notification(
                notification_id="notif-1",
                event=notification_event,
                channel=NotificationChannel.EMAIL,
                shadows=shadows
            )

        # Add second notification
        with patch.object(shadows, 'add', new_callable=AsyncMock):
            await batch_notification(
                notification_id="notif-2",
                event=notification_event,
                channel=NotificationChannel.EMAIL,
                shadows=shadows
            )

        assert len(notification_batches[batch_key]["notifications"]) == 2

    @pytest.mark.asyncio
    async def test_batch_notification_large_batch(self, shadows: Shadow, notification_event):
        """Test processing when batch reaches threshold."""
        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            # Add 5 notifications (triggers immediate processing)
            for i in range(5):
                await batch_notification(
                    notification_id=f"notif-{i}",
                    event=notification_event,
                    channel=NotificationChannel.EMAIL,
                    shadows=shadows
                )

            # Should trigger immediate batch processing
            assert mock_add.call_count >= 1
            assert "process_notification_batch" in str(mock_add.call_args)


class TestProcessNotificationBatch:
    """Test batch processing functionality."""

    @pytest.mark.asyncio
    async def test_process_notification_batch_single(self, shadows: Shadow, sample_batch):
        """Test processing a batch with single notification."""
        batch_key = "test-batch"
        notification_batches[batch_key] = sample_batch

        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            await process_notification_batch(batch_key, shadows)

            # Should deliver individual notification
            mock_add.assert_called_once()
            assert "deliver_notification" in str(mock_add.call_args)

            # Batch should be cleaned up
            assert batch_key not in notification_batches

    @pytest.mark.asyncio
    async def test_process_notification_batch_multiple(self, shadows: Shadow, notification_event):
        """Test processing a batch with multiple notifications."""
        batch_key = "test-batch-multi"
        notification_batches[batch_key] = {
            "notifications": [
                {
                    "id": "notif-1",
                    "event": notification_event,
                    "added_at": datetime.now(timezone.utc)
                },
                {
                    "id": "notif-2",
                    "event": notification_event,
                    "added_at": datetime.now(timezone.utc)
                }
            ],
            "created_at": datetime.now(timezone.utc),
            "channel": NotificationChannel.EMAIL
        }

        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            await process_notification_batch(batch_key, shadows)

            # Should deliver digest
            mock_add.assert_called_once()
            assert "deliver_notification_digest" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_process_notification_batch_missing(self, shadows: Shadow):
        """Test processing a non-existent batch."""
        with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
            await process_notification_batch("non-existent-batch", shadows)

            # Should log warning
            mock_logger.warning.assert_called_once()


class TestDeliverNotification:
    """Test notification delivery functionality."""

    @pytest.mark.asyncio
    async def test_deliver_email_notification(self, notification_event):
        """Test email notification delivery."""
        with patch('shadows.examples.real_time_notifications.print') as mock_print:
            await deliver_email_notification("notif-123", notification_event)

            mock_print.assert_called_once()
            assert "📧 EMAIL sent" in mock_print.call_args[0][0]

    @pytest.mark.asyncio
    async def test_deliver_sms_notification(self, notification_event):
        """Test SMS notification delivery."""
        with patch('shadows.examples.real_time_notifications.print') as mock_print:
            await deliver_sms_notification("notif-123", notification_event)

            mock_print.assert_called_once()
            assert "📱 SMS sent" in mock_print.call_args[0][0]

    @pytest.mark.asyncio
    async def test_deliver_push_notification(self, notification_event):
        """Test push notification delivery."""
        with patch('shadows.examples.real_time_notifications.print') as mock_print:
            await deliver_push_notification("notif-123", notification_event)

            mock_print.assert_called_once()
            assert "🔔 PUSH sent" in mock_print.call_args[0][0]

    @pytest.mark.asyncio
    async def test_deliver_webhook_notification(self, notification_event):
        """Test webhook notification delivery."""
        with patch('shadows.examples.real_time_notifications.print') as mock_print:
            await deliver_webhook_notification("notif-123", notification_event)

            mock_print.assert_called_once()
            assert "🔗 WEBHOOK sent" in mock_print.call_args[0][0]

    @pytest.mark.asyncio
    async def test_deliver_notification_with_retry(self, shadows: Shadow, notification_event):
        """Test notification delivery with retry on failure."""
        # Mock delivery to always fail
        with patch('shadows.examples.real_time_notifications.deliver_email_notification', side_effect=Exception("Delivery failed")):
            with patch('shadows.examples.real_time_notifications.track_delivery_attempt') as mock_track:
                # This should raise an exception to trigger retry
                with pytest.raises(Exception):
                    await deliver_notification(
                        notification_id="notif-123",
                        event=notification_event,
                        channel=NotificationChannel.EMAIL,
                        retry=ExponentialRetry(attempts=3, minimum_delay=timedelta(seconds=1))
                    )

                # Should track failed attempt
                mock_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_notification_success(self, shadows: Shadow, notification_event):
        """Test successful notification delivery."""
        with patch('shadows.examples.real_time_notifications.deliver_email_notification') as mock_deliver:
            with patch('shadows.examples.real_time_notifications.track_delivery_attempt') as mock_track:
                await deliver_notification(
                    notification_id="notif-123",
                    event=notification_event,
                    channel=NotificationChannel.EMAIL,
                    retry=ExponentialRetry(attempts=3, minimum_delay=timedelta(seconds=1))
                )

                # Should call delivery function
                mock_deliver.assert_called_once()

                # Should track successful delivery
                mock_track.assert_called_once()


class TestWebhookProcessing:
    """Test webhook processing functionality."""

    @pytest.mark.asyncio
    async def test_process_webhook_event_valid(self, shadows: Shadow, webhook_payload):
        """Test processing a valid webhook event."""
        with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
            with patch('shadows.examples.real_time_notifications.verify_webhook_signature', return_value=True):
                await process_webhook_event(webhook_payload, "valid_signature", shadows)

                # Should process the notification event
                mock_add.assert_called_once()
                assert "process_notification_event" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_process_webhook_event_invalid_signature(self, shadows: Shadow, webhook_payload):
        """Test processing webhook with invalid signature."""
        with patch('shadows.examples.real_time_notifications.verify_webhook_signature', return_value=False):
            with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
                await process_webhook_event(webhook_payload, "invalid_signature", shadows)

                # Should log warning about invalid signature
                mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_webhook_event_missing_fields(self, shadows: Shadow):
        """Test processing webhook with missing required fields."""
        invalid_payload = {"incomplete": "data"}

        with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
            await process_webhook_event(invalid_payload, "signature", shadows)

            # Should log error about missing fields
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_webhook_signature(self):
        """Test webhook signature verification."""
        # Valid signature
        assert await verify_webhook_signature({"test": "data"}, "valid_signature")

        # Invalid signature
        assert not await verify_webhook_signature({"test": "data"}, "invalid_signature")


class TestDeliveryTracking:
    """Test delivery tracking functionality."""

    @pytest.mark.asyncio
    async def test_track_delivery_attempt_success(self):
        """Test tracking successful delivery attempt."""
        with patch('shadows.examples.real_time_notifications.print') as mock_print:
            await track_delivery_attempt(
                notification_id="notif-123",
                channel=NotificationChannel.EMAIL,
                attempt=1,
                status="delivered"
            )

            # Should print tracking info
            mock_print.assert_called_once()
            assert "📊 Tracked delivery" in mock_print.call_args[0][0]
            assert "delivered" in mock_print.call_args[0][0]

    @pytest.mark.asyncio
    async def test_track_delivery_attempt_failed(self):
        """Test tracking failed delivery attempt."""
        with patch('shadows.examples.real_time_notifications.print') as mock_print:
            await track_delivery_attempt(
                notification_id="notif-123",
                channel=NotificationChannel.EMAIL,
                attempt=2,
                status="failed",
                error_message="Connection timeout"
            )

            # Should print tracking info with error
            mock_print.assert_called_once()
            assert "failed" in mock_print.call_args[0][0]


class TestNotificationDigest:
    """Test notification digest functionality."""

    @pytest.mark.asyncio
    async def test_deliver_notification_digest_email(self):
        """Test delivering email digest."""
        notifications = [
            {
                "event": NotificationEvent(
                    event_id="event-1",
                    user_id="user123",
                    event_type="test_event",
                    data={"key": "value"}
                )
            }
        ]

        with patch('shadows.examples.real_time_notifications.deliver_email_digest') as mock_deliver:
            await deliver_notification_digest(
                batch_key="test-batch",
                notifications=notifications,
                channel=NotificationChannel.EMAIL
            )

            mock_deliver.assert_called_once_with("user123", notifications)

    @pytest.mark.asyncio
    async def test_deliver_notification_digest_sms(self):
        """Test delivering SMS digest."""
        notifications = [
            {
                "event": NotificationEvent(
                    event_id="event-1",
                    user_id="user123",
                    event_type="test_event",
                    data={"key": "value"}
                )
            }
        ]

        with patch('shadows.examples.real_time_notifications.deliver_sms_digest') as mock_deliver:
            await deliver_notification_digest(
                batch_key="test-batch",
                notifications=notifications,
                channel=NotificationChannel.SMS
            )

            mock_deliver.assert_called_once_with("user123", notifications)


class TestProcessingNotification:
    """Test processing completion notifications."""

    @pytest.mark.asyncio
    async def test_send_processing_notification_completed(self):
        """Test sending completion notification."""
        # Mock file registry and processing results
        file_registry["file-123"] = MagicMock()
        file_registry["file-123"].uploaded_by = "user123"
        processing_results["file-123"] = MagicMock()
        processing_results["file-123"].success = True
        processing_results["file-123"].processing_time_seconds = 45.67

        with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
            await send_processing_notification("file-123", "completed")

            # Should log successful completion
            mock_logger.info.assert_called_once()
            assert "successfully" in str(mock_logger.info.call_args)

    @pytest.mark.asyncio
    async def test_send_processing_notification_failed(self):
        """Test sending failure notification."""
        file_registry["file-456"] = MagicMock()
        file_registry["file-456"].uploaded_by = "user456"
        processing_results["file-456"] = MagicMock()
        processing_results["file-456"].success = False
        processing_results["file-456"].error_message = "Processing failed"

        with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
            await send_processing_notification("file-456", "failed")

            # Should log failure
            mock_logger.error.assert_called_once()
            assert "failed" in str(mock_logger.error.call_args)


class TestCleanupFailedFile:
    """Test failed file cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_failed_file(self):
        """Test cleaning up files after processing failure."""
        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
                    await cleanup_failed_file("file-123", "/tmp/test.txt")

                    # Should attempt to remove file
                    mock_remove.assert_called_once_with("/tmp/test.txt")

                    # Should log cleanup completion
                    mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_failed_file_not_exists(self):
        """Test cleanup when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            with patch('os.remove') as mock_remove:
                with patch('shadows.examples.real_time_notifications.logger') as mock_logger:
                    await cleanup_failed_file("file-123", "/tmp/nonexistent.txt")

                    # Should not attempt to remove non-existent file
                    mock_remove.assert_not_called()

                    # Should still log completion
                    mock_logger.info.assert_called_once()
