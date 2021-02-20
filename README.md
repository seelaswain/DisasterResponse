# Disaster Response Pipeline Project
#1 Objective:Is to create a Machine Learning Pipeline to categorize messages into 36 different sections in real time.

#2 Partner:- Udacity for providing course content and guidance and Figure Eight for providing labelled dataset.

#3. This project has 3 key parts 
   - Process Data:- In this process we extracted, cleaned and saved data.
   - Build ML pipeline:- Train and test the dataset and classify messages into 36 different categories.
   - create and Run:- Here we created visualization to show output.

#4 Folder Structure
 - app ( templates(go.html,master.html), run.py)
 - data (process_data.py,DisasterResponse.csv,DisasterResponse.db,disaster_categories.csv,disaster_messages.csv)
 - models (classifier.pkl,train_classifier.py)
 
#5 Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
      
     Open another terminal and type     env|grep WORK
     
     https://SPACEID-3001.SPACEDOMAIN   (You will get spaceid and Spacedomain when you run env|grep WORK, replace those with spaceid and Spacedomain mentioned in sample URL )
     
 3. Go to https://0.0.0.0:3001/
 
#6 Acknowledgement:-

 1. Udacity Mentor support for guidance,improving performance,rectifying errors and clarifying doubts.
 2. Figure Eight for providing data set.
 3. https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv
 4. https://realpython.com/python-pep8/
 5. https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e
 6. https://www.youtube.com/watch?v=2vASHVT0qKc&t=269s, for github load
 