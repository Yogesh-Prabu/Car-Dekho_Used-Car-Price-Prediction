
# Car Dekho Price Prediction Project

## Project Overview
This project involves developing a predictive model for used car prices based on detailed attributes provided by the CarDekho dataset. The workflow spans from data structuring and cleaning through to feature engineering, preprocessing, visualization, and model training, culminating in a Flask web application for real-time predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation
To set up this project locally, you'll need Python and various libraries. Install them using pip:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn flask
```

## Usage
To get started with this project, follow these instructions:
1. Clone the repository:
   ```bash
   git clone [repository-link]
   ```
2. Navigate to the project's main directory:
   ```bash
   cd [repository-name]
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```

## Project Structure
- **01_data_structuring.ipynb** - Initializes the data environment by structuring raw data into a workable format for analysis.
- **02_cleaning.ipynb** - Cleans the dataset by handling missing values, outliers, and errors to prepare for accurate modeling.
- **03_feature_engineering.ipynb** - Develops new features that enhance the predictive power of the machine learning models based on the cleaned data.
- **04_preprocessing.ipynb** - Applies preprocessing techniques such as scaling and encoding to make the data suitable for training.
- **05_Visualizations.ipynb** - Generates insightful visualizations to explore data trends and relationships.
- **06_model_training.ipynb** - Focuses on training, tuning, and evaluating various machine learning models to find the best predictor for car prices.
- **app.py** - A Flask application that deploys the trained model to provide an interactive user interface for real-time price prediction.
- **optimized_random_forest_model.pkl** - The serialized version of the best-performing model ready for deployment.
- **PKL_Files/** - Contains additional model files and preprocessing transformers.
- **Raw_Datasets/** - Directory with the original datasets provided by CarDekho.
- **SBS_Processed_Datasets/** - Contains datasets that have been incrementally processed at each step of the project.

## Technologies Used
- **Python**: The primary programming language used.
- **Jupyter Notebook**: For creating and sharing documents that contain live code, equations, visualizations, and narrative text.
- **Libraries**: Utilizes Pandas for data manipulation, Scikit-Learn for machine learning, and Matplotlib/Seaborn for plotting.
- **Flask**: For creating the web application that utilizes our model for predictions.

## Contributing
Contributions to improve the project are welcome. Please fork the repository and create a pull request, or open an issue with the tags "enhancement" or "bug" to discuss potential changes.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments
- Heartfelt thanks to CarDekho for providing the dataset that made this project possible.
- Gratitude to all contributors who have participated in this project.
