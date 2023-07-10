# Project Downhill

## Description

Project Downhill is a web application that uses North American ski mountains and inputted zip codes to find and compare local resorts for users. The goal of the project is to provide a comprehensive tool for planning ski trips, taking into account the geographical location and preferences of the user.

### What is this for?

The project is a web application built with Python and Streamlit. It uses data scraped responsibly from onthesnow.com. It is comprised of North American ski mountains and user-inputted zip codes to find and compare local resorts. The project was created to help users plan their ski trips more efficiently.

### Usage examples

The main script of the project is `project_downhill.py`, which can be run to start the web application by running 'streamlit run project_downhill.py' in your Terminal or by visiting 'https://danmarino1-project-downhill-project-downhill-zzb2ku.streamlit.app'. The script includes data loading, data cleaning, some exploratory data analysis, and visualization of the ski resorts.

### Issues or limitations of the project

The data was collected from North American ski mountains in March of 2023, and any changes to their data could affect the functionality of the project. Additionally, the dataset is currently limited to North American resorts.

### Future features

Future updates may include analysis for different scenarios such as cost and snowfall and improvements to the data cleaning and analysis process. I will also be adding international support for mountains and zip codes.

## Technologies

The project is built with Python and uses the following libraries:

- pandas
- streamlit
- plotly
- numpy
- geopy

These libraries were chosen for their robust data processing, visualization capabilities, and ability to create interactive web applications.

## Details about use

To use the project, clone the repository and run the `project_downhill.py` script. The script will load the data, clean it, perform the analysis, and visualize the results in a Streamlit application.

Alternatively, visit 'https://danmarino1-project-downhill-project-downhill-zzb2ku.streamlit.app'. Note: the app may need to 'wake' since Streamlit only keeps applications readily available for one week.

### Contribution guidelines

Contributions are welcome. Please fork the repository and create a pull request with your changes. I'm eager to see what the community suggests!

### Credits

This project was created by danmarino1. I am grateful for the team at [www.onthesnow.com](https://www.onthesnow.com) for creating such a comprehensive dataset.

## Dependencies

The project has the following dependencies:

- pandas
- streamlit
- plotly
- numpy
- geopy

These can be installed by running `pip install -r requirements.txt` in your terminal.
