# Quick Setup Guide for Your Computer

## What You Need
- Python 3.11 or newer
- At least 4GB of free space
- Internet connection

## Step 1: Download the Project
1. Download all files from Replit
2. Create a new folder on your computer called `water-pump-prediction`
3. Put all files in this folder

## Step 2: Install Python Libraries
Open your terminal/command prompt and run:
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm catboost plotly matplotlib seaborn folium streamlit-folium pyyaml
```

## Step 3: Create the Right Folder Structure
Make sure your folder looks like this:
```
water-pump-prediction/
├── src/
│   ├── components/
│   ├── pipeline/
│   └── utils/
├── pages/
├── artifacts/
├── logs/
├── config/
├── sample_data/
├── uploaded_data/
└── app.py
```

## Step 4: Run the Application
In your terminal, go to the project folder and run:
```bash
streamlit run app.py
```

## Step 5: Open in Your Browser
The application will open automatically at `http://localhost:8501`

## What Each Part Does

### Main Files
- `app.py` - The main application you run
- `pages/` - Different sections of the web app
- `src/` - The brain of the system (machine learning code)

### Generated Files
- `artifacts/` - Saves your trained models
- `logs/` - Keeps track of what happened
- `uploaded_data/` - Stores files you upload

### How to Use
1. **Upload Data**: Use the upload section on the main page
2. **Explore Data**: Go to the EDA page to see your data
3. **Train Models**: Use the Model Training page
4. **Make Predictions**: Use the Predictions page
5. **See Maps**: Use the Geospatial Analysis page

## Common Problems and Solutions

### "Module not found" error
Run: `pip install [missing-module-name]`

### "Permission denied" error
- Make sure you have permission to write files in the folder
- Run terminal as administrator (Windows) or use `sudo` (Mac/Linux)

### App won't start
- Check if Python is installed: `python --version`
- Make sure you're in the right folder
- Try: `python -m streamlit run app.py`

### Out of memory
- Close other programs
- Use smaller data files for testing
- Restart your computer

## Need Help?
1. Check the logs folder for error details
2. Look at the full documentation in `PROJECT_DOCUMENTATION.md`
3. Make sure all files are in the right folders