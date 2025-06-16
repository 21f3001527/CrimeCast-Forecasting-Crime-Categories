# CrimeCast: Forecasting Crime Categories

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Crime%20Prediction-red.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)]()

## 🎯 Project Overview

**CrimeCast** is a comprehensive machine learning project focused on predicting crime categories using historical crime incident data. This project analyzes various aspects of criminal activities including location, timing, victim demographics, and incident characteristics to build accurate predictive models for law enforcement and public safety applications.

### 🔍 Problem Statement

The goal is to develop machine learning models capable of accurately predicting crime categories based on incident information. By leveraging data-driven insights, this project aims to:

- Enhance law enforcement strategies
- Improve resource allocation for crime prevention
- Bolster public safety measures through predictive analytics
- Transform raw crime data into actionable intelligence

## 📊 Dataset Information

### Dataset Overview
The dataset provides a comprehensive snapshot of criminal activities within the city, encompassing various aspects of each incident including date, time, location, victim demographics, and more.

### 📁 Data Files Structure
```
CrimeCast/
├── data/
│   ├── train.csv              # Training dataset with target variable
│   ├── test.csv               # Test dataset for predictions
│   └── sample_submission.csv  # Sample submission format
├── notebooks/
│   └── crime_prediction.ipynb # Main analysis notebook

└── README.md                 # Project documentation
```

### 🏷️ Dataset Features

| Feature | Description |
|---------|-------------|
| **Location** | Street address of the crime incident |
| **Cross_Street** | Cross street of the rounded address |
| **Latitude/Longitude** | Geographic coordinates of the incident |
| **Date_Reported** | Date the incident was reported |
| **Date_Occurred** | Date the incident occurred |
| **Time_Occurred** | Time of incident (24-hour military time) |
| **Area_ID** | LAPD's Geographic Area number |
| **Area_Name** | Name designation of the LAPD Geographic Area |
| **Reporting_District_no** | Reporting district number |
| **Part 1-2** | Crime classification |
| **Modus_Operandi** | Activities associated with the suspect |
| **Victim_Age** | Age of the victim |
| **Victim_Sex** | Gender of the victim |
| **Victim_Descent** | Descent code of the victim |
| **Premise_Code** | Premise code indicating location type |
| **Premise_Description** | Description of the premise code |
| **Weapon_Used_Code** | Weapon code indicating weapon type |
| **Weapon_Description** | Description of the weapon code |
| **Status** | Status of the case |
| **Status_Description** | Description of the status code |
| **Crime_Category** | 🎯 **Target Variable** - Category of the crime |

### 📈 Dataset Statistics
- **Size**: 5.79 MB
- **Format**: CSV files
- **Evaluation Metric**: Accuracy Score
- 
## 🛠️ Technologies Used

- **Python 3.8+** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment
- **Plotly** - Interactive visualizations (optional)

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/21f3001527/crimecast-prediction.git
   cd crimecast-prediction
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Run the Analysis**
   - Open `notebooks/crime_prediction.ipynb`
   - Execute all cells to reproduce the analysis

### Manual Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter plotly
```

## 📋 Project Workflow

### 1. **Data Exploration & Analysis**
- Dataset overview and statistical summary
- Missing value analysis
- Feature distribution analysis
- Temporal and geographical crime patterns

### 2. **Data Preprocessing**
- Handling missing values
- Feature engineering (time-based features, location clustering)
- Categorical encoding
- Feature scaling and normalization

### 3. **Feature Engineering**
- Time-based features (hour, day, month, season)
- Geographic clustering and area analysis
- Victim demographic patterns
- Crime location categorization

### 4. **Model Development**
- Multiple algorithm comparison
- Hyperparameter tuning
- Cross-validation
- Model evaluation and selection

### 5. **Model Evaluation**
- Accuracy assessment
- Confusion matrix analysis
- Feature importance analysis
- Performance visualization

## 🎯 Usage Example

```python
# Load the trained model and make predictions
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load test data
test_data = pd.read_csv('data/test.csv')

# Preprocess test data (same as training)
test_processed = preprocess_data(test_data)

# Make predictions
predictions = model.predict(test_processed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data.index,
    'crime_category': predictions
})
submission.to_csv('submission.csv', index=False)
```

## 📊 Key Insights and Findings

- **Temporal Patterns**: Analysis of crime occurrence by time of day, day of week, and seasonal trends
- **Geographic Hotspots**: Identification of high-crime areas and geographic clustering
- **Demographic Analysis**: Victim demographics and their correlation with crime types
- **Feature Importance**: Key factors that most influence crime category prediction

## 🏆 Model Performance

| Model | Accuracy | Performance Rank | Notes |
|-------|----------|------------------|-------|
| **RandomForest** | **95.0%** | 🥇 1st | Best performing model |
| **SVC** | **94.7%** | 🥈 2nd | Strong performance with support vectors |
| **Decision Tree** | **94.0%** | 🥉 3rd | Good interpretability |
| **AdaBoost** | **80.7%** | 4th | Ensemble boosting method |

### Key Performance Insights
- **RandomForest** achieved the highest accuracy of **95.0%**, making it the preferred model for crime category prediction
- **SVC (Support Vector Classifier)** performed exceptionally well with **94.7%** accuracy
- **Decision Tree** provided good performance with **94.0%** accuracy and excellent interpretability
- All top 3 models achieved over 94% accuracy, showing strong predictive capability

## 📁 File Structure

```
CrimeCast/
├── data/                     # Dataset files
├── crime_prediction.py       # Jupyter notebooks
└── README.md                 # Project documentation
```

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Implement neural networks for improved accuracy
- [ ] **Real-time Prediction**: Develop API for real-time crime prediction
- [ ] **Geographic Visualization**: Interactive crime mapping dashboard
- [ ] **Ensemble Methods**: Combine multiple models for better performance
- [ ] **Time Series Analysis**: Predict crime trends over time
- [ ] **External Data Integration**: Weather, events, economic indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rajeev Kumar**
- **Student ID**: 21f3001527
- **GitHub**: [@21f3001527](https://github.com/21f3001527)
- **LinkedIn**: [Rajeev Kumar](https://www.linkedin.com/in/rajeev245/)
- **Email**: 21f3001527@ds.study.iitm.ac.in
- **Institution**: Indian Institute of Technology Madras

## 🙏 Acknowledgments

- LAPD for making crime data publicly available
- Open source community for excellent machine learning libraries
- Fellow data scientists for insights and collaboration

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Crime Data Analysis Best Practices](https://example.com)

---

**⚡ Ready to dive into crime prediction?** Clone this repository and start exploring the fascinating world of predictive policing and public safety analytics!

**🎯 Challenge yourself:** Can you achieve higher accuracy than the baseline models? Fork this project and show us your skills!
