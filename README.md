# Panorama-Image-Stitching

## Usage
### Data Preparation (Optional)
  * Place your images under `$INDIR`
  * Run the following command, which will automatically read the focal lengths of images under `$INDIR` and create `focal_length.csv`

    ```bash
    cd code
    python3 read_focal_length.py $INDIR
    # E.g., 
    # python3 read_focal_length.py ../data/home
    ```
### Running the Code
* Command
    ```bash
    python main.py [--indir $INDIR] [--e2e] [--cache]
    # E.g.,
    # python3 main.py --indir ./data/parrington --cache
    ```
* Command-line Arguments
    ```bash
    --indir: Default is ../data/home.
    --e2e: Store-true. that the images are taken end-to-end.
    --cache: Store-true. Specifies whether to read preprocessed(cylindrical projected) images from cache.
    ```