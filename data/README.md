# Data Directory Structure

This directory contains all datasets used in the AIS D-STGT project. The data is organized following data science best practices.

## Directory Structure

```
data/
├── raw/           # Original, immutable data dump
├── interim/       # Intermediate data that has been transformed
├── processed/     # The final, canonical data sets for modeling
└── external/      # Data from third party sources
```

## Data Types

### Raw Data (`raw/`)
- Original AIS data files (CSV, JSON, etc.)
- Unprocessed vessel trajectory data
- Maritime traffic datasets
- **Never modify files in this directory**

### Interim Data (`interim/`)
- Partially cleaned and transformed data
- Data with basic preprocessing applied
- Temporary datasets during processing pipeline

### Processed Data (`processed/`)
- Final datasets ready for model training
- Cleaned and validated AIS trajectories
- Feature-engineered datasets
- Train/validation/test splits

### External Data (`external/`)
- Third-party datasets (NOAA, MarineCadastre, etc.)
- Reference data (ports, coastlines, etc.)
- Weather and oceanographic data
- Regulatory and compliance data

## Data Formats

- **CSV**: Tabular AIS data with timestamps
- **Parquet**: Optimized columnar format for large datasets
- **HDF5**: Hierarchical data for complex trajectory structures
- **GeoJSON**: Geographic data (ports, boundaries, etc.)

## Data Security & Compliance

- All AIS data must be **anonymized** before storage
- Follow **NOAA usage terms** for official datasets
- No sensitive vessel information in version control
- Use `.env` variables for data source credentials

## Usage Examples

```python
import pandas as pd
from pathlib import Path

# Load processed training data
data_dir = Path("data/processed")
train_data = pd.read_parquet(data_dir / "train_trajectories.parquet")

# Load external port data
external_dir = Path("data/external")
ports = pd.read_csv(external_dir / "world_ports.csv")
```

## Data Pipeline

1. **Raw** → Download and store original AIS data
2. **Interim** → Apply basic cleaning and validation
3. **Processed** → Feature engineering and final preparation
4. **External** → Integrate supplementary datasets

## File Naming Convention

- Use descriptive names: `vessel_trajectories_2023_01.csv`
- Include date ranges: `ais_data_20230101_20231231.parquet`
- Version datasets: `processed_trajectories_v1.2.parquet`
- Use lowercase with underscores: `san_francisco_bay_area.geojson`

## Data Validation

All processed datasets should include:
- Data quality reports
- Schema validation results
- Statistical summaries
- Missing value analysis

## Storage Considerations

- Large files (>100MB) should use `.gitignore`
- Consider data compression for archival
- Use cloud storage for very large datasets
- Maintain data lineage documentation
