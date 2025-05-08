# Data Directory

This project uses free portrait images downloaded from [Pixabay](https://pixabay.com/), which are licensed for free commercial and non-commercial use under the [Pixabay License](https://pixabay.com/service/license-summary/).

### Dataset Summary

- **Number of Images**:
  - Male: 472
  - Female: 471

- **Image Source**: All images were gathered from Pixabay to maintain consistency.

## How to Get the Data

1. Visit [Pixabay.com](https://pixabay.com/)
2. Search for royalty-free images using terms like “male portrait” and “female portrait”
3. Download and manually classify images into folders:

data/
├── female/
└── male/

**Note**: Due to licensing and storage limitations, the actual images are not included in this repository. You can recreate the dataset by searching and downloading gender-tagged portrait images from Pixabay directly.


## Preprocessing

- Resize all images to 130x130 pixels
- Normalize pixel values to [0, 1]
- Convert to RGB format if needed

You can modify `src/data_loader.py` to automatically handle these preprocessing steps during dataset loading.