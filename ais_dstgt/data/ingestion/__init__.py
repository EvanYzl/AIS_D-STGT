"""AIS Data Ingestion Module.

This module handles data ingestion from various sources including:
- Local CSV files
- NOAA HTML directory listings
- Remote data sources
- Database connections

The ingestion process includes:
- Data source discovery and validation
- Parallel data loading
- Initial data quality checks
- Metadata extraction
"""

from typing import List

__all__: list[str] = [
    "AISIngestionManager",
    "CSVIngestionHandler",
    "NOAAIngestionHandler",
]

# from .manager import AISIngestionManager
# from .csv_handler import CSVIngestionHandler
# from .noaa_handler import NOAAIngestionHandler
