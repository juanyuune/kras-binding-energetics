# Data

Download the experimental dataset from MaveDB:
- Accession: urn:mavedb:00000115
- URL: https://www.mavedb.org/scoresets/urn:mavedb:00000115/
- Citation: Weng et al. Nature 2024;626:643-652

Place downloaded files in this directory then run:
  python precompute/build_master_table.py \
    --input_dir data/raw/ \
    --output    data/kras_master_table.csv
