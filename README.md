# BeautiPho

## Overview
BeautiPho is a Python-based project designed for photo enhancement and beautification. It leverages image processing techniques to apply various filters and effects, enabling users to improve the visual quality of their photos effortlessly. This repository contains scripts and utilities to perform tasks such as skin whitening, color enhancement, and artistic effect application.

## Features
- **Skin Whitening**: Smooths and brightens skin tones for a polished look.
- **Color Filters**: Applies effects like HDR, retro, and LOMO to enhance photo aesthetics.
- **Border Effects**: Adds simple, textured, or ripped borders to images.
- **Customizable Settings**: Allows users to adjust effect intensity and other parameters.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/xzhang311/BeautiPho.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BeautiPho
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to process an image:
```bash
python abouthat.py --input path/to/image.jpg --output path/to/output.jpg --effect retro
```
Available effects include `retro`, `hdr`, `skin_whitening`, and more. Use the `--help` flag for detailed options:
```bash
python abouthat.py --help
```

## Requirements
- Python 3.8+
- Libraries: OpenCV, NumPy, Pillow (listed in `requirements.txt`)

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Inspired by open-source image processing tools like GIMP Beautify.[](https://github.com/hejiann/beautify)
