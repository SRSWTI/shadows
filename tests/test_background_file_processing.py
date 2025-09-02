"""Tests for background file processing example.

This module tests the file processing system's functionality including:
- File upload and validation workflows
- File chunking and parallel processing
- Progress tracking and status updates
- Error recovery and cleanup
- Different processing pipelines per file type
- Storage coordination and optimization
"""

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from uuid import uuid4

import pytest

from shadows import Shadow, Worker
from shadows.examples.background_file_processing import (
    FileType,
    ProcessingStatus,
    FileMetadata,
    ProcessingResult,
    file_registry,
    processing_results,
    initiate_file_upload,
    validate_file,
    chunk_file,
    process_file_chunk,
    reassemble_file_chunks,
    process_file,
    optimize_and_store_file,
    send_processing_notification,
    cleanup_failed_file,
    determine_file_type,
    get_max_file_size,
    get_chunk_size,
    calculate_file_checksum,
    validate_file_format,
    process_file_by_type,
    process_chunk_by_type,
    reassemble_chunks,
    optimize_file_by_type,
    store_file_final
)


@pytest.fixture
def sample_file_metadata():
    """Create sample file metadata for testing."""
    return FileMetadata(
        file_id="file-123",
        filename="test.jpg",
        original_name="test.jpg",
        file_type=FileType.IMAGE,
        mime_type="image/jpeg",
        size_bytes=1024 * 1024,  # 1MB
        uploaded_by="user123",
        uploaded_at=datetime.now(timezone.utc),
        checksum="abcd1234"
    )


@pytest.fixture
def large_file_metadata():
    """Create metadata for a large file that requires chunking."""
    return FileMetadata(
        file_id="file-large",
        filename="large_video.mp4",
        original_name="large_video.mp4",
        file_type=FileType.VIDEO,
        mime_type="video/mp4",
        size_bytes=100 * 1024 * 1024,  # 100MB
        uploaded_by="user456",
        uploaded_at=datetime.now(timezone.utc),
        checksum="large1234"
    )


@pytest.fixture(autouse)
def reset_globals():
    """Reset global state between tests."""
    file_registry.clear()
    processing_results.clear()


class TestFileMetadata:
    """Test FileMetadata dataclass."""

    def test_file_metadata_creation(self):
        """Test creating file metadata."""
        metadata = FileMetadata(
            file_id="test-123",
            filename="test.jpg",
            original_name="original.jpg",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
            size_bytes=2048,
            uploaded_by="user123",
            uploaded_at=datetime.now(timezone.utc),
            checksum="checksum123"
        )

        assert metadata.file_id == "test-123"
        assert metadata.filename == "test.jpg"
        assert metadata.file_type == FileType.IMAGE
        assert metadata.processing_status == ProcessingStatus.UPLOADED
        assert metadata.processing_progress == 0.0
        assert metadata.chunks == []
        assert metadata.output_files == []


class TestDetermineFileType:
    """Test file type determination."""

    @pytest.mark.parametrize("mime_type,file_path,expected", [
        ("image/jpeg", "test.jpg", FileType.IMAGE),
        ("video/mp4", "test.mp4", FileType.VIDEO),
        ("application/pdf", "test.pdf", FileType.DOCUMENT),
        ("audio/mp3", "test.mp3", FileType.AUDIO),
        ("application/zip", "test.zip", FileType.ARCHIVE),
        ("unknown/type", "test.unknown", FileType.DOCUMENT),  # Default
    ])
    def test_determine_file_type(self, mime_type, file_path, expected):
        """Test file type determination from MIME type and path."""
        result = determine_file_type(mime_type, file_path)
        assert result == expected

    def test_determine_file_type_by_extension(self):
        """Test file type determination by file extension."""
        assert determine_file_type("", "image.png") == FileType.IMAGE
        assert determine_file_type("", "video.avi") == FileType.VIDEO
        assert determine_file_type("", "doc.docx") == FileType.DOCUMENT
        assert determine_file_type("", "audio.flac") == FileType.AUDIO
        assert determine_file_type("", "archive.tar.gz") == FileType.ARCHIVE


class TestFileSizeLimits:
    """Test file size limit functions."""

    @pytest.mark.parametrize("file_type,expected_max", [
        (FileType.IMAGE, 100 * 1024 * 1024),      # 100MB
        (FileType.VIDEO, 2 * 1024 * 1024 * 1024), # 2GB
        (FileType.AUDIO, 500 * 1024 * 1024),      # 500MB
        (FileType.DOCUMENT, 50 * 1024 * 1024),    # 50MB
        (FileType.ARCHIVE, 1 * 1024 * 1024 * 1024) # 1GB
    ])
    def test_get_max_file_size(self, file_type, expected_max):
        """Test maximum file size limits per file type."""
        assert get_max_file_size(file_type) == expected_max

    @pytest.mark.parametrize("file_type,expected_chunk", [
        (FileType.IMAGE, 10 * 1024 * 1024),    # 10MB
        (FileType.VIDEO, 50 * 1024 * 1024),    # 50MB
        (FileType.AUDIO, 25 * 1024 * 1024),    # 25MB
        (FileType.DOCUMENT, 5 * 1024 * 1024),  # 5MB
        (FileType.ARCHIVE, 100 * 1024 * 1024)  # 100MB
    ])
    def test_get_chunk_size(self, file_type, expected_chunk):
        """Test chunk size per file type."""
        assert get_chunk_size(file_type) == expected_chunk


class TestCalculateFileChecksum:
    """Test file checksum calculation."""

    @pytest.mark.asyncio
    async def test_calculate_file_checksum(self):
        """Test SHA256 checksum calculation."""
        test_data = b"Hello, World!"
        test_file = "/tmp/test_checksum.txt"

        # Create test file
        with open(test_file, "wb") as f:
            f.write(test_data)

        try:
            checksum = await calculate_file_checksum(test_file)

            # Should be a valid SHA256 hash (64 characters, hex)
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

            # Same content should produce same checksum
            checksum2 = await calculate_file_checksum(test_file)
            assert checksum == checksum2

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    @pytest.mark.asyncio
    async def test_calculate_file_checksum_nonexistent(self):
        """Test checksum calculation for non-existent file."""
        with pytest.raises(FileNotFoundError):
            await calculate_file_checksum("/nonexistent/file.txt")


class TestValidateFileFormat:
    """Test file format validation."""

    @pytest.mark.asyncio
    async def test_validate_file_format_image(self, sample_file_metadata):
        """Test image file format validation."""
        with patch('os.path.exists', return_value=True):
            # Should not raise for valid image format
            await validate_file_format("/tmp/test.jpg", sample_file_metadata)

    @pytest.mark.asyncio
    async def test_validate_file_format_invalid_image(self):
        """Test invalid image file format validation."""
        metadata = FileMetadata(
            file_id="test",
            filename="test.txt",
            original_name="test.txt",
            file_type=FileType.IMAGE,
            mime_type="text/plain",
            size_bytes=1024,
            uploaded_by="user",
            uploaded_at=datetime.now(timezone.utc),
            checksum="test"
        )

        with pytest.raises(ValueError, match="Invalid image format"):
            await validate_file_format("/tmp/test.txt", metadata)


class TestInitiateFileUpload:
    """Test file upload initiation."""

    @pytest.mark.asyncio
    async def test_initiate_file_upload_success(self, shadows: Shadow):
        """Test successful file upload initiation."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"test image data" * 100)  # ~1.6KB
            temp_path = temp_file.name

        try:
            with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                file_id = await initiate_file_upload(temp_path, "user123", shadows)

                # Should generate a file ID
                assert file_id.startswith("file-")
                assert len(file_id) > 5

                # Should store metadata in registry
                assert file_id in file_registry
                metadata = file_registry[file_id]
                assert metadata.uploaded_by == "user123"
                assert metadata.file_type == FileType.IMAGE
                assert metadata.processing_status == ProcessingStatus.UPLOADED

                # Should start validation process
                mock_add.assert_called_once()
                assert "validate_file" in str(mock_add.call_args)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @pytest.mark.asyncio
    async def test_initiate_file_upload_metadata(self, shadows: Shadow):
        """Test that file upload creates correct metadata."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"test pdf content" * 50)
            temp_path = temp_file.name

        try:
            with patch.object(shadows, 'add', new_callable=AsyncMock):
                file_id = await initiate_file_upload(temp_path, "user456", shadows)

                metadata = file_registry[file_id]
                assert metadata.filename == os.path.basename(temp_path)
                assert metadata.original_name == os.path.basename(temp_path)
                assert metadata.file_type == FileType.DOCUMENT
                assert metadata.mime_type == "application/pdf"
                assert metadata.size_bytes > 0
                assert metadata.checksum is not None
                assert len(metadata.checksum) == 64  # SHA256 hex length

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestValidateFile:
    """Test file validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_file_success(self, shadows: Shadow, sample_file_metadata):
        """Test successful file validation."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata

        with patch('os.path.exists', return_value=True):
            with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                await validate_file(
                    sample_file_metadata.file_id,
                    "/tmp/test.jpg",
                    shadows
                )

                # Should update status
                assert sample_file_metadata.processing_status == ProcessingStatus.VALIDATING

                # Should proceed to processing (not chunking for small file)
                mock_add.assert_called_once()
                assert "process_file" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_validate_file_too_large(self, shadows: Shadow):
        """Test validation of oversized file."""
        large_metadata = FileMetadata(
            file_id="large-file",
            filename="large.mp4",
            original_name="large.mp4",
            file_type=FileType.VIDEO,
            mime_type="video/mp4",
            size_bytes=3 * 1024 * 1024 * 1024,  # 3GB (over limit)
            uploaded_by="user",
            uploaded_at=datetime.now(timezone.utc),
            checksum="test"
        )
        file_registry[large_metadata.file_id] = large_metadata

        with patch('os.path.exists', return_value=True):
            with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                await validate_file(large_metadata.file_id, "/tmp/large.mp4", shadows)

                # Should mark as failed
                assert large_metadata.processing_status == ProcessingStatus.FAILED
                assert "too large" in large_metadata.error_message.lower()

                # Should trigger cleanup
                mock_add.assert_called_once()
                assert "cleanup_failed_file" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_validate_file_not_exists(self, shadows: Shadow, sample_file_metadata):
        """Test validation of non-existent file."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata

        with patch('os.path.exists', return_value=False):
            with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                await validate_file(
                    sample_file_metadata.file_id,
                    "/nonexistent/file.jpg",
                    shadows
                )

                # Should mark as failed
                assert sample_file_metadata.processing_status == ProcessingStatus.FAILED
                assert "does not exist" in sample_file_metadata.error_message

    @pytest.mark.asyncio
    async def test_validate_file_checksum_mismatch(self, shadows: Shadow, sample_file_metadata):
        """Test validation with checksum mismatch."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata
        sample_file_metadata.checksum = "wrong_checksum"

        with patch('os.path.exists', return_value=True):
            with patch('shadows.examples.background_file_processing.calculate_file_checksum', return_value="correct_checksum"):
                with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                    await validate_file(
                        sample_file_metadata.file_id,
                        "/tmp/test.jpg",
                        shadows
                    )

                    # Should mark as failed due to checksum mismatch
                    assert sample_file_metadata.processing_status == ProcessingStatus.FAILED
                    assert "checksum mismatch" in sample_file_metadata.error_message


class TestChunkFile:
    """Test file chunking functionality."""

    @pytest.mark.asyncio
    async def test_chunk_file_large(self, shadows: Shadow, large_file_metadata):
        """Test chunking a large file."""
        file_registry[large_file_metadata.file_id] = large_file_metadata

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=b"x" * (100 * 1024 * 1024))):
                with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                    await chunk_file(
                        large_file_metadata.file_id,
                        "/tmp/large_video.mp4",
                        shadows
                    )

                    # Should update status
                    assert large_file_metadata.processing_status == ProcessingStatus.CHUNKING

                    # Should create chunks
                    assert len(large_file_metadata.chunks) > 0

                    # Should schedule chunk processing for each chunk
                    assert mock_add.call_count == len(large_file_metadata.chunks) + 1  # chunks + reassembly

                    # Should schedule reassembly
                    reassemble_calls = [call for call in mock_add.call_args_list if "reassemble_file_chunks" in str(call)]
                    assert len(reassemble_calls) == 1

    @pytest.mark.asyncio
    async def test_chunk_file_metadata(self, shadows: Shadow, large_file_metadata):
        """Test that chunking creates correct metadata."""
        file_registry[large_file_metadata.file_id] = large_file_metadata

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=b"x" * (100 * 1024 * 1024))):
                with patch.object(shadows, 'add', new_callable=AsyncMock):
                    await chunk_file(
                        large_file_metadata.file_id,
                        "/tmp/large_video.mp4",
                        shadows
                    )

                    # Verify chunk metadata structure
                    for chunk in large_file_metadata.chunks:
                        assert "chunk_number" in chunk
                        assert "start_offset" in chunk
                        assert "size_bytes" in chunk
                        assert "file_path" in chunk
                        assert "status" in chunk
                        assert chunk["status"] == "created"


class TestProcessFile:
    """Test file processing functionality."""

    @pytest.mark.asyncio
    async def test_process_file_success(self, shadows: Shadow, sample_file_metadata):
        """Test successful file processing."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata

        with patch('shadows.examples.background_file_processing.process_file_by_type', return_value=b"processed_data"):
            with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                await process_file(
                    sample_file_metadata.file_id,
                    "/tmp/test.jpg",
                    shadows
                )

                # Should update status and progress
                assert sample_file_metadata.processing_status == ProcessingStatus.PROCESSING
                assert sample_file_metadata.processing_progress == 100.0

                # Should create output file metadata
                assert len(sample_file_metadata.output_files) == 1
                output_file = sample_file_metadata.output_files[0]
                assert output_file["type"] == "processed"
                assert "path" in output_file
                assert "size_bytes" in output_file
                assert "created_at" in output_file

                # Should proceed to optimization
                mock_add.assert_called_once()
                assert "optimize_and_store_file" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_process_file_failure(self, shadows: Shadow, sample_file_metadata):
        """Test file processing failure."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata

        with patch('shadows.examples.background_file_processing.process_file_by_type', side_effect=Exception("Processing failed")):
            with patch('shadows.examples.background_file_processing.logger') as mock_logger:
                await process_file(
                    sample_file_metadata.file_id,
                    "/tmp/test.jpg",
                    shadows
                )

                # Should mark as failed
                assert sample_file_metadata.processing_status == ProcessingStatus.FAILED
                assert "Processing failed" in sample_file_metadata.error_message

                # Should log error
                mock_logger.error.assert_called_once()


class TestOptimizeAndStoreFile:
    """Test file optimization and storage."""

    @pytest.mark.asyncio
    async def test_optimize_and_store_success(self, shadows: Shadow, sample_file_metadata):
        """Test successful file optimization and storage."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata
        sample_file_metadata.output_files = [{"type": "processed", "path": "/tmp/processed.jpg"}]

        with patch('shadows.examples.background_file_processing.optimize_file_by_type'):
            with patch('shadows.examples.background_file_processing.store_file_final', return_value="/storage/final/file.jpg"):
                with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                    await optimize_and_store_file(sample_file_metadata.file_id, shadows)

                    # Should update status and timestamps
                    assert sample_file_metadata.processing_status == ProcessingStatus.COMPLETED
                    assert sample_file_metadata.processed_at is not None

                    # Should create processing result
                    assert sample_file_metadata.file_id in processing_results
                    result = processing_results[sample_file_metadata.file_id]
                    assert result.success is True
                    assert result.processing_time_seconds > 0

                    # Should send completion notification
                    mock_add.assert_called_once()
                    assert "send_processing_notification" in str(mock_add.call_args)

    @pytest.mark.asyncio
    async def test_optimize_and_store_failure(self, shadows: Shadow, sample_file_metadata):
        """Test file optimization/storage failure."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata

        with patch('shadows.examples.background_file_processing.optimize_file_by_type', side_effect=Exception("Optimization failed")):
            with patch('shadows.examples.background_file_processing.logger') as mock_logger:
                with patch.object(shadows, 'add', new_callable=AsyncMock) as mock_add:
                    await optimize_and_store_file(sample_file_metadata.file_id, shadows)

                    # Should mark as failed
                    assert sample_file_metadata.processing_status == ProcessingStatus.FAILED
                    assert "Optimization failed" in sample_file_metadata.error_message

                    # Should send failure notification
                    mock_add.assert_called_once()
                    assert "send_processing_notification" in str(mock_add.call_args)


class TestSendProcessingNotification:
    """Test processing completion notifications."""

    @pytest.mark.asyncio
    async def test_send_completion_notification(self, sample_file_metadata):
        """Test sending successful completion notification."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata
        processing_results[sample_file_metadata.file_id] = ProcessingResult(
            file_id=sample_file_metadata.file_id,
            success=True,
            output_files=[],
            processing_time_seconds=30.5
        )

        with patch('shadows.examples.background_file_processing.logger') as mock_logger:
            await send_processing_notification(sample_file_metadata.file_id, "completed")

            mock_logger.info.assert_called_once()
            call_args = str(mock_logger.info.call_args)
            assert "completed successfully" in call_args
            assert "30.5" in call_args

    @pytest.mark.asyncio
    async def test_send_failure_notification(self, sample_file_metadata):
        """Test sending failure notification."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata
        processing_results[sample_file_metadata.file_id] = ProcessingResult(
            file_id=sample_file_metadata.file_id,
            success=False,
            output_files=[],
            processing_time_seconds=15.2,
            error_message="Processing failed"
        )

        with patch('shadows.examples.background_file_processing.logger') as mock_logger:
            await send_processing_notification(sample_file_metadata.file_id, "failed")

            mock_logger.error.assert_called_once()
            call_args = str(mock_logger.error.call_args)
            assert "failed" in call_args
            assert "Processing failed" in call_args


class TestCleanupFailedFile:
    """Test failed file cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_with_existing_files(self, sample_file_metadata):
        """Test cleanup when files exist."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata
        sample_file_metadata.chunks = [
            {"file_path": "/tmp/chunk1.dat"},
            {"file_path": "/tmp/chunk2.dat"}
        ]
        sample_file_metadata.output_files = [
            {"path": "/tmp/output1.jpg"},
            {"path": "/tmp/output2.jpg"}
        ]

        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                with patch('shadows.examples.background_file_processing.logger') as mock_logger:
                    await cleanup_failed_file(
                        sample_file_metadata.file_id,
                        "/tmp/original.jpg"
                    )

                    # Should remove original file
                    assert mock_remove.call_count >= 1
                    remove_calls = [str(call) for call in mock_remove.call_args_list]
                    assert any("/tmp/original.jpg" in call for call in remove_calls)

                    # Should remove chunks
                    assert any("/tmp/chunk1.dat" in call for call in remove_calls)
                    assert any("/tmp/chunk2.dat" in call for call in remove_calls)

                    # Should remove output files
                    assert any("/tmp/output1.jpg" in call for call in remove_calls)
                    assert any("/tmp/output2.jpg" in call for call in remove_calls)

                    # Should log completion
                    mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_missing_files(self, sample_file_metadata):
        """Test cleanup when some files don't exist."""
        file_registry[sample_file_metadata.file_id] = sample_file_metadata
        sample_file_metadata.chunks = [{"file_path": "/nonexistent/chunk.dat"}]

        with patch('os.path.exists', return_value=False):
            with patch('os.remove') as mock_remove:
                with patch('shadows.examples.background_file_processing.logger') as mock_logger:
                    await cleanup_failed_file(
                        sample_file_metadata.file_id,
                        "/nonexistent/original.jpg"
                    )

                    # Should not attempt to remove non-existent files
                    mock_remove.assert_not_called()

                    # Should still log completion
                    mock_logger.info.assert_called_once()


class TestProcessingFunctions:
    """Test file processing utility functions."""

    @pytest.mark.asyncio
    async def test_process_file_by_type_image(self):
        """Test image file processing."""
        with patch('shadows.examples.background_file_processing.process_image', return_value=b"processed_image"):
            result = await process_file_by_type("/tmp/test.jpg", FileType.IMAGE)
            assert result == b"processed_image"

    @pytest.mark.asyncio
    async def test_process_file_by_type_video(self):
        """Test video file processing."""
        with patch('shadows.examples.background_file_processing.process_video', return_value=b"processed_video"):
            result = await process_file_by_type("/tmp/test.mp4", FileType.VIDEO)
            assert result == b"processed_video"

    @pytest.mark.asyncio
    async def test_process_chunk_by_type(self):
        """Test chunk processing."""
        result = await process_chunk_by_type("/tmp/chunk.dat", FileType.IMAGE)
        assert "processed_chunk_" in result

    @pytest.mark.asyncio
    async def test_reassemble_chunks(self, sample_file_metadata):
        """Test chunk reassembly."""
        with patch('shadows.examples.background_file_processing.logger'):
            result = await reassemble_chunks(sample_file_metadata)
            assert "reassembled" in result
            assert sample_file_metadata.filename.split('.')[-1] in result

    @pytest.mark.asyncio
    async def test_optimize_file_by_type(self, sample_file_metadata):
        """Test file optimization."""
        # Should complete without error
        await optimize_file_by_type(sample_file_metadata)

    @pytest.mark.asyncio
    async def test_store_file_final(self, sample_file_metadata):
        """Test final file storage."""
        result = await store_file_final(sample_file_metadata)
        assert "/storage/final/" in result
        assert sample_file_metadata.file_id in result
