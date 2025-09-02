#!/usr/bin/env python3
"""
Background file processing example demonstrating media processing pipelines with Shadows.

This example shows how to build a comprehensive file processing system that handles:
- File upload and validation
- Media processing (images, videos, documents)
- File chunking for large files
- Parallel processing of chunks
- Progress tracking and status updates
- Error recovery and cleanup
- Different processing pipelines per file type
- Storage coordination and optimization
- Completion notifications
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

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


class FileType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    AUDIO = "audio"
    ARCHIVE = "archive"


class ProcessingStatus(Enum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    CHUNKING = "chunking"
    PROCESSING = "processing"
    OPTIMIZING = "optimizing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileMetadata:
    file_id: str
    filename: str
    original_name: str
    file_type: FileType
    mime_type: str
    size_bytes: int
    uploaded_by: str
    uploaded_at: datetime
    checksum: str
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    processing_progress: float = 0.0
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None
    output_files: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessingResult:
    file_id: str
    success: bool
    output_files: List[Dict[str, Any]]
    processing_time_seconds: float
    error_message: Optional[str] = None


# Global file storage (in production, use database/redis)
file_registry = {}
processing_results = {}


async def initiate_file_upload(
    file_path: str,
    uploaded_by: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> str:
    """
    Initiate file upload processing workflow.

    Demonstrates:
    - File metadata extraction
    - Task workflow initiation
    - Initial validation setup
    """
    logger.info("initiating file upload", extra={"file_path": file_path, "uploaded_by": uploaded_by})

    # Generate unique file ID
    file_id = f"file-{hashlib.md5(f'{file_path}-{uploaded_by}-{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"

    # Extract file metadata
    file_stat = os.stat(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)

    # Determine file type
    file_type = determine_file_type(mime_type or "", file_path)

    # Create file metadata
    metadata = FileMetadata(
        file_id=file_id,
        filename=os.path.basename(file_path),
        original_name=os.path.basename(file_path),
        file_type=file_type,
        mime_type=mime_type or "application/octet-stream",
        size_bytes=file_stat.st_size,
        uploaded_by=uploaded_by,
        uploaded_at=datetime.now(),
        checksum=await calculate_file_checksum(file_path)
    )

    # Store metadata
    file_registry[file_id] = metadata

    logger.info("file metadata created", extra={
        "file_id": file_id,
        "file_type": file_type.value,
        "size_bytes": metadata.size_bytes
    })

    # Start processing workflow
    await shadows.add(validate_file)(
        file_id=file_id,
        file_path=file_path,
        key=f"validate-{file_id}"
    )

    return file_id


async def validate_file(
    file_id: str,
    file_path: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Validate uploaded file integrity and format.

    Demonstrates:
    - File integrity validation
    - Format verification
    - Size and security checks
    - Error handling and cleanup
    """
    metadata = file_registry.get(file_id)
    if not metadata:
        logger.error("file metadata not found", extra={"file_id": file_id})
        return

    metadata.processing_status = ProcessingStatus.VALIDATING
    logger.info("validating file", extra={"file_id": file_id, "file_path": file_path})

    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")

        # Validate file size
        max_size = get_max_file_size(metadata.file_type)
        if metadata.size_bytes > max_size:
            raise ValueError(f"File too large: {metadata.size_bytes} > {max_size}")

        # Validate file format
        await validate_file_format(file_path, metadata)

        # Validate checksum
        calculated_checksum = await calculate_file_checksum(file_path)
        if calculated_checksum != metadata.checksum:
            raise ValueError("File checksum mismatch - possible corruption")

        logger.info("file validation successful", extra={"file_id": file_id})

        # Proceed to chunking or direct processing
        if metadata.size_bytes > 50 * 1024 * 1024:  # 50MB threshold
            await shadows.add(chunk_file)(
                file_id=file_id,
                file_path=file_path,
                key=f"chunk-{file_id}"
            )
        else:
            await shadows.add(process_file)(
                file_id=file_id,
                file_path=file_path,
                key=f"process-{file_id}"
            )

    except Exception as e:
        logger.error("file validation failed", extra={"file_id": file_id, "error": str(e)})
        metadata.processing_status = ProcessingStatus.FAILED
        metadata.error_message = str(e)

        # Cleanup invalid file
        await shadows.add(cleanup_failed_file)(
            file_id=file_id,
            file_path=file_path,
            key=f"cleanup-{file_id}"
        )


async def chunk_file(
    file_id: str,
    file_path: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Split large files into manageable chunks for parallel processing.

    Demonstrates:
    - File chunking strategies
    - Parallel processing setup
    - Progress tracking
    """
    metadata = file_registry.get(file_id)
    if not metadata:
        return

    metadata.processing_status = ProcessingStatus.CHUNKING
    logger.info("chunking large file", extra={"file_id": file_id, "size_bytes": metadata.size_bytes})

    try:
        # Determine chunk size based on file type
        chunk_size = get_chunk_size(metadata.file_type)
        total_chunks = (metadata.size_bytes + chunk_size - 1) // chunk_size

        logger.info("creating file chunks", extra={
            "file_id": file_id,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size
        })

        # Create chunks
        chunks = []
        with open(file_path, 'rb') as f:
            for chunk_num in range(total_chunks):
                chunk_start = chunk_num * chunk_size
                chunk_data = f.read(chunk_size)

                if chunk_data:
                    chunk_file_path = f"{file_path}.chunk_{chunk_num:04d}"
                    async with await asyncio.get_event_loop().run_in_executor(None, open, chunk_file_path, 'wb') as chunk_file:
                        chunk_file.write(chunk_data)

                    chunk_info = {
                        "chunk_number": chunk_num,
                        "start_offset": chunk_start,
                        "size_bytes": len(chunk_data),
                        "file_path": chunk_file_path,
                        "status": "created"
                    }
                    chunks.append(chunk_info)

                    # Process chunk in parallel
                    await shadows.add(process_file_chunk)(
                        file_id=file_id,
                        chunk_info=chunk_info,
                        key=f"process-chunk-{file_id}-{chunk_num}"
                    )

        metadata.chunks = chunks
        logger.info("file chunking completed", extra={
            "file_id": file_id,
            "chunks_created": len(chunks)
        })

        # Schedule chunk reassembly after all chunks are processed
        await shadows.add(reassemble_file_chunks)(
            file_id=file_id,
            key=f"reassemble-{file_id}",
            when=datetime.now() + timedelta(minutes=5)  # Give time for processing
        )

    except Exception as e:
        logger.error("file chunking failed", extra={"file_id": file_id, "error": str(e)})
        metadata.processing_status = ProcessingStatus.FAILED
        metadata.error_message = str(e)


async def process_file_chunk(
    file_id: str,
    chunk_info: Dict[str, Any],
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Process individual file chunks.

    Demonstrates:
    - Parallel chunk processing
    - Progress tracking
    - Error handling per chunk
    """
    chunk_num = chunk_info["chunk_number"]
    logger.info("processing file chunk", extra={
        "file_id": file_id,
        "chunk_number": chunk_num,
        "chunk_size": chunk_info["size_bytes"]
    })

    try:
        chunk_info["status"] = "processing"
        chunk_info["processing_started"] = datetime.now()

        # Process chunk based on file type
        metadata = file_registry[file_id]
        result = await process_chunk_by_type(chunk_info["file_path"], metadata.file_type)

        chunk_info["status"] = "completed"
        chunk_info["processing_completed"] = datetime.now()
        chunk_info["result"] = result

        logger.info("chunk processing completed", extra={
            "file_id": file_id,
            "chunk_number": chunk_num,
            "result": result
        })

    except Exception as e:
        logger.error("chunk processing failed", extra={
            "file_id": file_id,
            "chunk_number": chunk_num,
            "error": str(e)
        })
        chunk_info["status"] = "failed"
        chunk_info["error"] = str(e)


async def reassemble_file_chunks(
    file_id: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Reassemble processed chunks into final output.

    Demonstrates:
    - Chunk reassembly logic
    - Validation of all chunks
    - Final processing completion
    """
    metadata = file_registry.get(file_id)
    if not metadata:
        return

    logger.info("reassembling file chunks", extra={
        "file_id": file_id,
        "total_chunks": len(metadata.chunks)
    })

    try:
        # Check if all chunks are processed
        failed_chunks = [c for c in metadata.chunks if c["status"] == "failed"]
        if failed_chunks:
            raise ValueError(f"Some chunks failed: {len(failed_chunks)}")

        completed_chunks = [c for c in metadata.chunks if c["status"] == "completed"]
        if len(completed_chunks) != len(metadata.chunks):
            # Not all chunks completed yet, reschedule
            logger.info("not all chunks completed, rescheduling", extra={
                "file_id": file_id,
                "completed": len(completed_chunks),
                "total": len(metadata.chunks)
            })
            await shadows.add(reassemble_file_chunks)(
                file_id=file_id,
                key=f"reassemble-{file_id}",
                when=datetime.now() + timedelta(minutes=1)
            )
            return

        # All chunks completed, reassemble
        output_file = await reassemble_chunks(metadata)
        metadata.output_files.append({
            "type": "processed",
            "path": output_file,
            "size_bytes": os.path.getsize(output_file),
            "created_at": datetime.now()
        })

        logger.info("file reassembly completed", extra={"file_id": file_id})

        # Final optimization and storage
        await shadows.add(optimize_and_store_file)(
            file_id=file_id,
            key=f"optimize-{file_id}"
        )

    except Exception as e:
        logger.error("file reassembly failed", extra={"file_id": file_id, "error": str(e)})
        metadata.processing_status = ProcessingStatus.FAILED
        metadata.error_message = str(e)


async def process_file(
    file_id: str,
    file_path: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Process entire file (for smaller files that don't need chunking).

    Demonstrates:
    - Direct file processing
    - Type-specific processing logic
    - Progress updates
    """
    metadata = file_registry.get(file_id)
    if not metadata:
        return

    metadata.processing_status = ProcessingStatus.PROCESSING
    start_time = datetime.now()

    logger.info("processing file", extra={
        "file_id": file_id,
        "file_type": metadata.file_type.value,
        "size_bytes": metadata.size_bytes
    })

    try:
        # Process based on file type
        result = await process_file_by_type(file_path, metadata.file_type)

        # Create output file info
        output_file = f"{file_path}.processed"
        metadata.output_files.append({
            "type": "processed",
            "path": output_file,
            "size_bytes": len(result) if isinstance(result, bytes) else 0,
            "created_at": datetime.now()
        })

        processing_time = (datetime.now() - start_time).total_seconds()
        metadata.processing_progress = 100.0

        logger.info("file processing completed", extra={
            "file_id": file_id,
            "processing_time_seconds": processing_time
        })

        # Move to optimization phase
        await shadows.add(optimize_and_store_file)(
            file_id=file_id,
            key=f"optimize-{file_id}"
        )

    except Exception as e:
        logger.error("file processing failed", extra={"file_id": file_id, "error": str(e)})
        metadata.processing_status = ProcessingStatus.FAILED
        metadata.error_message = str(e)


async def optimize_and_store_file(
    file_id: str,
    shadows: Shadow = CurrentShadow(),
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Optimize processed file and store in final location.

    Demonstrates:
    - File optimization techniques
    - Storage coordination
    - Final processing completion
    """
    metadata = file_registry.get(file_id)
    if not metadata:
        return

    metadata.processing_status = ProcessingStatus.OPTIMIZING
    logger.info("optimizing and storing file", extra={"file_id": file_id})

    try:
        # Optimize based on file type
        await optimize_file_by_type(metadata)

        # Store in final location
        final_path = await store_file_final(metadata)
        metadata.processing_status = ProcessingStatus.STORING

        # Update metadata
        metadata.processed_at = datetime.now()
        metadata.processing_status = ProcessingStatus.COMPLETED

        # Store results
        processing_results[file_id] = ProcessingResult(
            file_id=file_id,
            success=True,
            output_files=metadata.output_files,
            processing_time_seconds=(metadata.processed_at - metadata.uploaded_at).total_seconds()
        )

        logger.info("file processing completed successfully", extra={
            "file_id": file_id,
            "processing_time_seconds": processing_results[file_id].processing_time_seconds,
            "output_files_count": len(metadata.output_files)
        })

        # Send completion notification
        await shadows.add(send_processing_notification)(
            file_id=file_id,
            status="completed",
            key=f"notify-{file_id}"
        )

    except Exception as e:
        logger.error("file optimization/storage failed", extra={"file_id": file_id, "error": str(e)})
        metadata.processing_status = ProcessingStatus.FAILED
        metadata.error_message = str(e)

        processing_results[file_id] = ProcessingResult(
            file_id=file_id,
            success=False,
            output_files=[],
            processing_time_seconds=(datetime.now() - metadata.uploaded_at).total_seconds(),
            error_message=str(e)
        )

        # Send failure notification
        await shadows.add(send_processing_notification)(
            file_id=file_id,
            status="failed",
            key=f"notify-{file_id}"
        )


async def send_processing_notification(
    file_id: str,
    status: str,
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Send notification about processing completion/failure.

    Demonstrates:
    - Notification integration
    - Status reporting
    """
    metadata = file_registry.get(file_id)
    result = processing_results.get(file_id)

    if status == "completed":
        logger.info("file processing completed successfully", extra={
            "file_id": file_id,
            "user": metadata.uploaded_by if metadata else "unknown",
            "processing_time_seconds": result.processing_time_seconds if result else 0
        })
        print(f"✅ File {file_id} processing completed in {result.processing_time_seconds:.2f}s")
    else:
        logger.error("file processing failed", extra={
            "file_id": file_id,
            "error": result.error_message if result else "unknown error"
        })
        print(f"❌ File {file_id} processing failed: {result.error_message if result else 'unknown error'}")


async def cleanup_failed_file(
    file_id: str,
    file_path: str,
    logger: logging.LoggerAdapter[logging.Logger] = TaskLogger()
) -> None:
    """
    Clean up files after processing failure.

    Demonstrates:
    - Resource cleanup
    - Error recovery
    """
    logger.info("cleaning up failed file", extra={"file_id": file_id, "file_path": file_path})

    try:
        # Remove original file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove any chunks
        metadata = file_registry.get(file_id)
        if metadata and metadata.chunks:
            for chunk in metadata.chunks:
                chunk_path = chunk.get("file_path")
                if chunk_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)

        # Remove any output files
        if metadata and metadata.output_files:
            for output_file in metadata.output_files:
                output_path = output_file.get("path")
                if output_path and os.path.exists(output_path):
                    os.remove(output_path)

        logger.info("cleanup completed", extra={"file_id": file_id})

    except Exception as e:
        logger.error("cleanup failed", extra={"file_id": file_id, "error": str(e)})


# Utility functions

def determine_file_type(mime_type: str, file_path: str) -> FileType:
    """Determine file type from MIME type and file extension."""
    if mime_type.startswith("image/"):
        return FileType.IMAGE
    elif mime_type.startswith("video/"):
        return FileType.VIDEO
    elif mime_type.startswith("audio/"):
        return FileType.AUDIO
    elif mime_type in ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument"]:
        return FileType.DOCUMENT
    elif mime_type in ["application/zip", "application/x-tar", "application/gzip"]:
        return FileType.ARCHIVE

    # Fallback to extension
    ext = Path(file_path).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        return FileType.IMAGE
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return FileType.VIDEO
    elif ext in [".mp3", ".wav", ".flac"]:
        return FileType.AUDIO
    elif ext in [".pdf", ".doc", ".docx", ".txt"]:
        return FileType.DOCUMENT
    elif ext in [".zip", ".tar", ".gz"]:
        return FileType.ARCHIVE

    return FileType.DOCUMENT  # Default


def get_max_file_size(file_type: FileType) -> int:
    """Get maximum file size for file type."""
    sizes = {
        FileType.IMAGE: 100 * 1024 * 1024,      # 100MB
        FileType.VIDEO: 2 * 1024 * 1024 * 1024, # 2GB
        FileType.AUDIO: 500 * 1024 * 1024,      # 500MB
        FileType.DOCUMENT: 50 * 1024 * 1024,    # 50MB
        FileType.ARCHIVE: 1 * 1024 * 1024 * 1024 # 1GB
    }
    return sizes.get(file_type, 100 * 1024 * 1024)


def get_chunk_size(file_type: FileType) -> int:
    """Get chunk size for file type."""
    sizes = {
        FileType.IMAGE: 10 * 1024 * 1024,    # 10MB
        FileType.VIDEO: 50 * 1024 * 1024,    # 50MB
        FileType.AUDIO: 25 * 1024 * 1024,    # 25MB
        FileType.DOCUMENT: 5 * 1024 * 1024,  # 5MB
        FileType.ARCHIVE: 100 * 1024 * 1024  # 100MB
    }
    return sizes.get(file_type, 10 * 1024 * 1024)


async def calculate_file_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


async def validate_file_format(file_path: str, metadata: FileMetadata) -> None:
    """Validate file format based on type."""
    # Mock validation - in production would check file signatures, headers, etc.
    await asyncio.sleep(0.1)

    if metadata.file_type == FileType.IMAGE:
        if not metadata.mime_type.startswith("image/"):
            raise ValueError("Invalid image format")
    elif metadata.file_type == FileType.VIDEO:
        if not metadata.mime_type.startswith("video/"):
            raise ValueError("Invalid video format")


async def process_file_by_type(file_path: str, file_type: FileType) -> Any:
    """Process file based on type."""
    await asyncio.sleep(0.5)  # Mock processing time

    if file_type == FileType.IMAGE:
        return await process_image(file_path)
    elif file_type == FileType.VIDEO:
        return await process_video(file_path)
    elif file_type == FileType.DOCUMENT:
        return await process_document(file_path)
    elif file_type == FileType.AUDIO:
        return await process_audio(file_path)
    else:
        return await process_generic(file_path)


async def process_chunk_by_type(chunk_path: str, file_type: FileType) -> Any:
    """Process file chunk based on type."""
    await asyncio.sleep(0.3)  # Mock processing time
    return f"processed_chunk_{Path(chunk_path).name}"


async def reassemble_chunks(metadata: FileMetadata) -> str:
    """Reassemble processed chunks."""
    await asyncio.sleep(0.2)  # Mock reassembly time
    return f"{metadata.file_id}_reassembled.{metadata.filename.split('.')[-1]}"


async def optimize_file_by_type(metadata: FileMetadata) -> None:
    """Optimize file based on type."""
    await asyncio.sleep(0.1)  # Mock optimization


async def store_file_final(metadata: FileMetadata) -> str:
    """Store file in final location."""
    await asyncio.sleep(0.05)  # Mock storage
    return f"/storage/final/{metadata.file_id}"


# Mock processing functions
async def process_image(file_path: str) -> bytes:
    return b"processed_image_data"

async def process_video(file_path: str) -> bytes:
    return b"processed_video_data"

async def process_document(file_path: str) -> bytes:
    return b"processed_document_data"

async def process_audio(file_path: str) -> bytes:
    return b"processed_audio_data"

async def process_generic(file_path: str) -> bytes:
    return b"processed_generic_data"


async def main():
    """Demonstrate the file processing system with various scenarios."""
    async with run_redis("7.4.2") as redis_url:
        async with Shadow(name="file-processing", url=redis_url) as shadows:
            # Register all file processing tasks
            shadows.register(initiate_file_upload)
            shadows.register(validate_file)
            shadows.register(chunk_file)
            shadows.register(process_file_chunk)
            shadows.register(reassemble_file_chunks)
            shadows.register(process_file)
            shadows.register(optimize_and_store_file)
            shadows.register(send_processing_notification)
            shadows.register(cleanup_failed_file)

            print("=== Background File Processing Demo ===\n")

            # Create some test files
            test_files = await create_test_files()

            # Scenario 1: Small image file (direct processing)
            print("1. Processing small image file...")
            file_id_1 = await initiate_file_upload(
                test_files["small_image"],
                "user123"
            )

            # Scenario 2: Large video file (chunked processing)
            print("\n2. Processing large video file (will be chunked)...")
            file_id_2 = await initiate_file_upload(
                test_files["large_video"],
                "user456"
            )

            # Scenario 3: Document file
            print("\n3. Processing document file...")
            file_id_3 = await initiate_file_upload(
                test_files["document"],
                "user789"
            )

            print("\n=== Starting File Processing Workers ===")
            async with Worker(shadows, concurrency=8) as worker:
                await worker.run_until_finished()

            print("\n=== File Processing Complete ===")
            print(f"Processed files: {len(file_registry)}")
            print(f"Successful: {len([f for f in processing_results.values() if f.success])}")
            print(f"Failed: {len([f for f in processing_results.values() if not f.success])}")

            # Show results
            for file_id, result in processing_results.items():
                metadata = file_registry[file_id]
                print(f"\n📁 {file_id}:")
                print(f"   Status: {'✅' if result.success else '❌'} {metadata.processing_status.value}")
                print(".2f")
                if result.output_files:
                    print(f"   Output files: {len(result.output_files)}")
                if result.error_message:
                    print(f"   Error: {result.error_message}")


async def create_test_files() -> Dict[str, str]:
    """Create test files for demonstration."""
    test_files = {}

    # Small image file
    test_files["small_image"] = "/tmp/test_small_image.jpg"
    with open(test_files["small_image"], "wb") as f:
        f.write(b"x" * (1024 * 1024))  # 1MB

    # Large video file (will trigger chunking)
    test_files["large_video"] = "/tmp/test_large_video.mp4"
    with open(test_files["large_video"], "wb") as f:
        f.write(b"x" * (100 * 1024 * 1024))  # 100MB

    # Document file
    test_files["document"] = "/tmp/test_document.pdf"
    with open(test_files["document"], "wb") as f:
        f.write(b"x" * (2 * 1024 * 1024))  # 2MB

    return test_files


if __name__ == "__main__":
    asyncio.run(main())
