# Image Recovery
Given an input image and a directory containing images, this application finds the N images most similar to the input image, with N being a number entered by the user.

## Getting started

### How to run the application:
In the terminal, clone the project:
> git clone https://github.com/andersonsantos3/image_recovery.git

After downloading, inside the project folder, run the following command:
> python3 main.py -h

This command will return the positional and optional arguments to run the application. They are:
#### Positional arguments
- input_image: Path of image to be compared.
- search_path: Path to search for images to compare.

#### Optional arguments
- number_of_matches: Number of matches to be returned. Default = 10.
- image_extensions: Image extensions allowed. Default = [jpg, png]. Type new image extensions to add.
- save: Saves the graphics and results if a path is entered.

The Vistex folder contained in this project's directory has several images. An example of a command to run the application is:
> python3 main.py Vistex/c001_001.png ./Vistex --number_of_images 5 --save ./results

This command will open the image c_001_001.png and search the Vistex folder for images similar to it. For this, the histograms will be calculated and from them the Probability Density Function (PDF). The distance between PDFs will be measured using Euclidean Distance. At the end, the 5 images that have the shortest distances will be displayed on the screen and saved in the results folder.

If you do not want to save the resulting images, simply do not enter the --save parameter.
