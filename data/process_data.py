'''Purpose:-Load and clean Databse
Pre-requisite Messages.csv and Categories.csv dataset available
To run ETL pipeline that cleans data and stores in database run the following in the project's root directory:-
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`'''

#1. Load libraries.
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


#2. Load Data
def load_data(messages_filepath, categories_filepath):
    ''' In this section of the code we read following inputs
    1) messages 2) categories and produced a dataframe(df) by merging both by id'''
  
    #Read data
    messages= pd.read_csv(messages_filepath)
    categories= pd.read_csv(categories_filepath)
   
    #Merge messages and categories dataset abd return merged data in a dataframe
    df = pd.merge(messages, categories, on='id')
    return df


#3 Clean Data
def clean_data(df):
    '''This section of code takes the merged dataset in a dataframe and produces cleaned dataset'''
    
    categories = df["categories"].str.split(";",expand=True)
    #To define the column names of categories,extract the first row of categories dataset
    row = categories.iloc[0]
    # Use the above generated row to extract a list of new column names for categories, use lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # Rename the columns of `categories 
    categories.columns = category_colnames
      
  

    #Convert category values to just numbers 0 or 1

    for column in categories:
        #Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
       #While doing data wrangling in Jupyternotebook I found that "related" column value 0,1,2. As we need only 0 &1 hence replacing 2 with 1.
        categories.related.replace(2,1,inplace=True)
        
  
    # Drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df


#4. Save Data
def save_data(df, database_filename):
    
     '''Save database.Always use function here, I hardcoded the database name befor which produced error in run.py stage. 
       Based on the guidance by mentor, changed it to below function..'''
        
     engine = create_engine('sqlite:///'+database_filename)
     print(df.head())
     df.to_sql('DisasterResponse', engine, index=False, if_exists='replace') 
    
    
            
#5 Execute your program             
def main():
    '''The main function in Python acts as the point of execution for any program. Defining the main function in Python 
        programming is a necessity to start the execution of the program as it gets executed.'''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()