# Data Availability

Due to the large size of the input datasets and licensing restrictions (?) associated with some data providers, the raw data used in this project are not distributed as part of this repository. 

All raw datasets can be retrieved directly from their original sources as described below.  Some input datasets have to be retrieved manually from indicated sources as of May 2026.

For the data retrieval pipeline to work properly, place each raw data file in its corresponding folder under `artifacts/data_raw`.

## Mosquito observation datasets

In this study, we used two types of mosquito observation datasets. The first dataset consists of historical mosquito observations obtained from:

* **Dataset name**: The global compendium of Aedes aegypti and Ae. albopictus occurrence
* **Data DOI**: [https://doi.org/10.5061/dryad.47v3c](https://doi.org/10.5061/dryad.47v3c) 
It provides mosquito occurrence records on a yearly basis. Observation locations are represented as polygons with side lengths ranging from approximately 5 km to 110 km, reflecting varying levels of spatial precision in the reported observations. 

The historical observation dataset was used as a spatial mask for the extraction of environmental and land-use suitability parameters, which served as input features for training the base model.

Please note, these data are also referred to in another publication: [http://dx.doi.org/10.7554/eLife.08347](http://dx.doi.org/10.7554/eLife.08347)

For the incremental learning component, mosquito observation data obtained from the: Global Mosquito Observations Dashboard (GMOD) were used.
* **Dataset name**: Global Mosquito Observations Dashboard (GMOD)
* **Data URL**: [GMOD](https://www.mosquitodashboard.org/)

**Important:** Unfortunately, following a recent update of the platform, direct download of the dataset is no longer available. Researchers seeking to reproduce or extend this work are encouraged to use comparable mosquito occurrence datasets from sources such as Global Biodiversity Information Facility (GBIF), iNaturalist, or Mosquito Alert, which provide similar georeferenced mosquito observation records suitable for incremental learning applications.

## Climate Data

The climate data used in this project can be obtained from:

* **Dataset name**: ERA5 monthly averaged data on single levels from 1940 to present
* **Data DOI**: [10.24381/cds.f17050d7](https://doi.org/10.24381/cds.f17050d7)

This dataset contains monthly mean values of 2 m air temperature, 2 m dew point temperature, total precipitation, and 10 m wind speed on a 0.25° spatial grid from 1940 to the present. It is available from the Copernicus Climate Data Store ERA5 reanalysis dataset. More information can be found at the Climate Data Store website.

For this study, we used data covering the period 1975–2024.

After downloading, place the dataset files in the `artifacts/data_raw/ERA5` directory and update the corresponding filename in the `config.py` file.

**Important:** In some cases, the ERA5 data provider may split the requested data into multiple files. The current data processing workflow does not support multiple input files automatically. If multiple files are provided, they must be merged before processing, or the workflow code must be modified accordingly.

## Land Use / Land Cover (LULC) Data

The land use and land cover data used in this project can be obtained from:

* **Dataset name**: HILDA+ Global land use change (version 1.0): HILDAplus_vGLOB-1.0-f
* **Data DOI**: [https://doi.org/10.1594/PANGAEA.921846 ](https://doi.org/10.1594/PANGAEA.921846)

The HILDA+ vGLOB-v1.0 dataset can be downloaded from PANGAEA. It provides annual global land-cover data at a spatial resolution of 0.01°. For this study, we used the GeoTIFF (.tif) version of the dataset, which contains eight major land-cover classes - ocean, urban, cropland, pasture, forest, grass/shrub, sparse vegetation, and water - for the period 1960–2019.

After downloading, all files must be extracted from the archives and placed in the `artifacts/data_raw/HILDA+` directory. 

## Population Data

The population data used in this project can be obtained from:

* **Dataset name**: Global Human Settlement Layer population grid (R2023) (GHS-POP_GLOBE_R2023A)
* **URL**: [GHS-POP_GLOBE_R2023A](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop)

Global Human Settlement Layer population data were downloaded as GHS-POP dataset from the Joint Research Centre of the European Commission. The GHS-POP dataset is provided as GeoTIFF files, one for each epoch/year: 1975, 1980, ..., 2030. For this study, the dataset was downloaded in the WGS84 coordinate reference system (EPSG:4326) at a spatial resolution of 30 arcseconds.

After downloading, all files must be extracted from the archives and placed in the `artifacts/data_raw/GHSL_POP` directory. 
