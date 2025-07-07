"""CSV Data Ingestion Handler.

This module handles loading and initial processing of AIS data from CSV files.
It provides efficient loading with validation and error handling.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..schemas import AISDataFrame

logger = logging.getLogger(__name__)


class CSVIngestionHandler:
    """Handler for CSV file ingestion with validation and error handling."""

    def __init__(
        self,
        chunk_size: int = 10000,
        validate_on_load: bool = True,
        handle_encoding_errors: bool = True,
    ):
        """Initialize CSV handler.

        Args:
            chunk_size: Size of chunks for processing large files
            validate_on_load: Whether to validate data during loading
            handle_encoding_errors: Whether to handle encoding errors gracefully
        """
        self.chunk_size = chunk_size
        self.validate_on_load = validate_on_load
        self.handle_encoding_errors = handle_encoding_errors
        self.supported_encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

    def load_file(
        self, file_path: str | Path, encoding: str | None = None, **kwargs
    ) -> AISDataFrame:
        """Load a single CSV file.

        Args:
            file_path: Path to the CSV file
            encoding: File encoding (auto-detected if None)
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            AISDataFrame: Validated AIS data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading CSV file: {file_path}")

        # Auto-detect encoding if not provided
        if encoding is None:
            encoding = self._detect_encoding(file_path)

        try:
            # Load data with pandas
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)

            logger.info(f"Loaded {len(df)} records from {file_path}")

            # Validate and return wrapped DataFrame
            if self.validate_on_load:
                return AISDataFrame(df)
            else:
                return df

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def load_files(
        self, file_paths: list[str | Path], combine: bool = True, **kwargs
    ) -> AISDataFrame | list[AISDataFrame]:
        """Load multiple CSV files.

        Args:
            file_paths: List of file paths
            combine: Whether to combine all files into one DataFrame
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            AISDataFrame or List[AISDataFrame]: Loaded data
        """
        logger.info(f"Loading {len(file_paths)} CSV files")

        dataframes = []

        for file_path in tqdm(file_paths, desc="Loading CSV files"):
            try:
                df = self.load_file(file_path, **kwargs)
                if self.validate_on_load:
                    dataframes.append(df.df)
                else:
                    dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        if not dataframes:
            raise ValueError("No files were successfully loaded")

        if combine:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(
                f"Combined {len(dataframes)} files into {len(combined_df)} records"
            )
            return AISDataFrame(combined_df)
        else:
            return [AISDataFrame(df) for df in dataframes]

    def load_chunked(
        self, file_path: str | Path, encoding: str | None = None, **kwargs
    ) -> pd.io.parsers.TextFileReader:
        """Load file in chunks for memory-efficient processing.

        Args:
            file_path: Path to the CSV file
            encoding: File encoding (auto-detected if None)
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            TextFileReader: Chunked reader object
        """
        file_path = Path(file_path)

        if encoding is None:
            encoding = self._detect_encoding(file_path)

        return pd.read_csv(
            file_path, encoding=encoding, chunksize=self.chunk_size, **kwargs
        )

    def get_file_info(self, file_path: str | Path) -> dict:
        """Get basic information about a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dict: File information including size, encoding, columns
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file size
        file_size = file_path.stat().st_size

        # Detect encoding
        encoding = self._detect_encoding(file_path)

        # Read first few rows to get column info
        try:
            sample_df = pd.read_csv(file_path, encoding=encoding, nrows=5)
            columns = sample_df.columns.tolist()
            dtypes = sample_df.dtypes.to_dict()
        except Exception as e:
            logger.warning(f"Could not read sample from {file_path}: {e}")
            columns = []
            dtypes = {}

        return {
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "encoding": encoding,
            "columns": columns,
            "dtypes": {k: str(v) for k, v in dtypes.items()},
        }

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding by trying common encodings.

        Args:
            file_path: Path to the file

        Returns:
            str: Detected encoding
        """
        if not self.handle_encoding_errors:
            return "utf-8"

        # Try to read first few lines with different encodings
        for encoding in self.supported_encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    # Try to read first 5 lines
                    for _ in range(5):
                        line = f.readline()
                        if not line:
                            break
                logger.debug(f"Detected encoding {encoding} for {file_path}")
                return encoding
            except UnicodeDecodeError:
                continue

        # If all fail, default to utf-8 with error handling
        logger.warning(f"Could not detect encoding for {file_path}, using utf-8")
        return "utf-8"

    def validate_csv_structure(self, file_path: str | Path) -> dict:
        """Validate CSV file structure and content.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dict: Validation results
        """
        file_path = Path(file_path)

        try:
            # Get basic file info
            info = self.get_file_info(file_path)

            # Check required columns
            required_cols = AISDataFrame.REQUIRED_COLUMNS
            missing_cols = set(required_cols) - set(info["columns"])

            # Load sample data for validation
            sample_df = pd.read_csv(file_path, encoding=info["encoding"], nrows=100)

            validation_results = {
                "file_path": str(file_path),
                "is_valid": len(missing_cols) == 0,
                "missing_required_columns": list(missing_cols),
                "total_columns": len(info["columns"]),
                "sample_records": len(sample_df),
                "encoding": info["encoding"],
                "file_size_mb": info["file_size_mb"],
            }

            # Additional validation checks
            if validation_results["is_valid"]:
                # Check for empty values in required columns
                empty_required = {}
                for col in required_cols:
                    if col in sample_df.columns:
                        empty_count = sample_df[col].isnull().sum()
                        if empty_count > 0:
                            empty_required[col] = empty_count

                validation_results["empty_required_fields"] = empty_required
                validation_results["has_empty_required"] = len(empty_required) > 0

            return validation_results

        except Exception as e:
            return {"file_path": str(file_path), "is_valid": False, "error": str(e)}
