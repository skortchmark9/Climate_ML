# Project 2: Climate Predictions with Machine Learning

## Predicting the Ocean Surface Boundary Layer's Density Profiles using Observational Data 

### Group 7: Azam Khan, Samuel Kortchmar, Ahinoam Toubia

## Overview
In this project, our group apply the same methodology from the paper *Sane et al. (2023) "Parameterizing vertical mixing coefficients in the ocean surface boundary layer using neural networks."* to estimate density profiles using observational data from Papa Station (Located in the North Pacific Ocean, 50° North latitude, 145° West longitude) and analyze the effects of seasonality on the dataset.

### Research Question:
**Attempt to predict density & diffusivity of the mixed layer from observational data, with examination of the influence of seasonality.**

### Tasks Addressed:
- **Data Processing and Preparation**: Acquire and clean data from Papa Station as well as reading and filtering the Generalized Ocean Turbulence Model (GOTM) output.
- **Identify the depths of the mixed layer**: Determine the mixed layer depths (MLD), $h$, from Papa Station Mooring Data.
- **Neural Network 1 Training**: Implementing a neural network to predict a density profile from SST, SSS, and time.
- **Neural Network 2 Training**: Implementing a neural network to predict diffusivity from density profiles, MLD, and time
- **Model Evaluation and Visualization**: Evaluate the models' performance using metrics such as training and validation loss, and present the results visually.
- **Paper vs. observational data**: Examine the relationship between observational data variables and the input variables used in the paper.
- **Seasonality**: Explain the effects of seasonality on GOTM simulations vs. observational data.

## Contribution Statment:
Sam and Azam clean Papa Station data (https://www.pmel.noaa.gov/ocs/data/disdel/). Azam create a NN which takes as inputs: (SST, SSS, and time) to predict density. Sam used the observational data from papa station and ploted the density overtime. Sam also managed to create a diffusivity profile based on the observational data. Ahinoam wrote up the explanations for the graph and the text for the data story, focusing on the climate science and seasonality, as well as explaining how the observational data variables relate to the input variables used in the paper. Azam and Sam carried out the computation for model evaluation. All team members contributed to the GitHub repository and prepared the presentation. All team members approve our work presented in our GitHub repository including this contribution statement.


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/skortchmark9/Climate_ML.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Climate_ML/Project_2
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Clone the repository
2. Navigate to the project directory
3. Run the cells in the `main.ipynb` Jupyter notebook to generate the results
    

## Project Structure
```
Climate_ML/
├── data/               # Raw and processed data
├── main.ipynb          # Jupyter notebook for submission
├── utils.py            # Utility functions
├── predict_diffusivity.py # Predict diffusivity
├── process_sim_data.py # Process simulation data
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

