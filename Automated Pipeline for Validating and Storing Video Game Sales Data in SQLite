#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
from faker import Faker
import random
import sqlite3
import pandas as pd


# In[181]:


df = pd.read_csv('./vgsales.csv/vgsales.csv')


# In[182]:


df


# In[183]:


df['Year'] = df['Year'].fillna(0).astype(int)
df


# ## Convert to String 

# In[184]:


df = df.astype(str)


# In[185]:


df.info()


# ## Create Rules (with regex)

# In[9]:




def validate_data(data):
    
    for index, row in data.iterrows():
        # Validate Year
        if not re.match(r'^\d{4}$', str(row['Year'])):
            print(f"Invalid Year at index {index}: {row['Year']}")

        # Validate Sales Figures
        if not re.match(r'^(\d+|\d*\.\d+)$', str(row['NA_Sales'])):
            print(f"Invalid NA_Sales at index {index}: {row['NA_Sales']}")
            
        if not re.match(r'^(\d+|\d*\.\d+)$', str(row['EU_Sales'])):
            print(f"Invalid EU_Sales at index {index}: {row['EU_Sales']}")
            
        if not re.match(r'^(\d+|\d*\.\d+)$', str(row['JP_Sales'])):
            print(f"Invalid JP_Sales at index {index}: {row['JP_Sales']}")
            
        if not re.match(r'^(\d+|\d*\.\d+)$', str(row['Other_Sales'])):
            print(f"Invalid Other_Sales at index {index}: {row['Other_Sales']}")
            
        if not re.match(r'^(\d+|\d*\.\d+)$', str(row['Global_Sales'])):
            print(f"Invalid Global_Sales at index {index}: {row['Global_Sales']}")

        # Validate Names
        if not re.match(r'^.+$', str(row['Name'])):
            print(f"Invalid Game Name at index {index}: {row['Name']}")
            
        if not re.match(r'^.+$', str(row['Platform'])):
            print(f"Invalid Platform Name at index {index}: {row['Platform']}")
            
        if not re.match(r'^[A-Za-z0-9 .:&\'-]+$', str(row['Genre'])):
            print(f"Invalid Genre Name at index {index}: {row['Genre']}")
            
        if not re.match(r'^.+$', str(row['Publisher'])):
            print(f"Invalid Publisher Name at index {index}: {row['Publisher']}")


# In[187]:


validate_data(df)


# In[125]:


## this mean that all names are allowed as long as it is not empty
## check if all sales are numbers/decimal numbers only
## check is all Years are four digit numbers


# ## Create Random Synthetic Data

# In[10]:


fake = Faker()
def create_synthetic_data():
    for _ in range(100):
        # Choose occasionally to include invalid or strange inputs
        year = str(fake.year())
        if random.choice([True, False]):  # Randomly decide to corrupt the year
            year = year.replace("1", "!", 1) if "1" in year else year + random.choice(["!", "@", "#"])
        
        genre = fake.random_element(elements=('Action', 'Adventure', 'Strategy'))
        if random.choice([True, False]):  # Randomly add special characters
            genre += random.choice(["!", "@", "#", "%%"])

        publisher = fake.company()
        if random.choice([True, False]):  # Randomly insert spaces 
            publisher = " " + publisher + " "

        # For sales
        na_sales = str(fake.random_number(digits=2))
        eu_sales = str(fake.random_number(digits=2))
        jp_sales = str(fake.random_number(digits=2))
        other_sales = str(fake.random_number(digits=2))
        global_sales = str(fake.random_number(digits=2))

        if random.choice([True, False]):
            na_sales = " " + na_sales  # Leading space
            eu_sales = eu_sales + "%"  # Special character
            jp_sales = "None"  # Non-numeric
            other_sales = other_sales.replace(other_sales[0], "x", 1) if other_sales else other_sales  # Replace first digit
            global_sales = global_sales + " "  

        yield {
            'Name': fake.text(max_nb_chars=20).strip(),
            'Platform': fake.random_element(elements=('PC', 'PS4', 'Xbox One','P S4')),
            'Year': year,
            'Genre': genre,
            'Publisher': publisher,
            'NA_Sales': na_sales,
            'EU_Sales': eu_sales,
            'JP_Sales': jp_sales,
            'Other_Sales': other_sales,
            'Global_Sales': global_sales
        }


# Convert the generator output to a DataFrame
df1 = pd.DataFrame(list(create_synthetic_data()))


# In[11]:


df1


# In[12]:


validate_data(df1)


# ## Creating a database

# In[188]:


conn = sqlite3.connect('games.db')
c = conn.cursor()


# In[189]:


c.execute("""
CREATE TABLE platforms
(platform_id INTEGER PRIMARY KEY AUTOINCREMENT,
platform TEXT UNIQUE)
""")
c.execute("""
CREATE TABLE genres
(genres_id INTEGER PRIMARY KEY AUTOINCREMENT,
genre TEXT UNIQUE)
""")
c.execute("""
CREATE TABLE publishers
(publishers_id INTEGER PRIMARY KEY AUTOINCREMENT,
publisher TEXT UNIQUE)
""")
c.execute("""
CREATE TABLE games
(games_id INTEGER PRIMARY KEY AUTOINCREMENT,
genre_id INTEGER,
name TEXT, year TEXT,
FOREIGN KEY(genre_id) REFERENCES genres(genre_id) )
""")
c.execute("""
CREATE TABLE sales
(sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
game_id INTEGER, platform_id INTEGER, publisher_id INTEGER,
na_sales REAL, eu_sales REAL, jp_sales REAL, other_sales REAL, global_sales REAL, 
FOREIGN KEY(game_id) REFERENCES games(game_id), 
FOREIGN KEY(platform_id) REFERENCES platforms(platform_id),
FOREIGN KEY(publisher_id) REFERENCES publishers(publisher_id))
""")


# In[190]:


c.execute("""
SELECT name FROM sqlite_master WHERE type='table'
""")

tables = c.fetchall()

for table in tables:
    table_name = table[0]
    print(f"\nTable: {table_name}")
    
    c.execute(f"PRAGMA table_info('{table_name}')")
    columns = c.fetchall()
    for column in columns:
        print(f"Column: {column[1]} | Type: {column[2]}")


# In[191]:


pd.read_sql("""
SELECT *
FROM sales
""", conn)


# ## Ingesting the Data to SQl

# In[192]:


df.columns = df.columns.str.strip()

# Helper function to insert and retrieve IDs from reference tables
def get_or_create(cursor, table, column, value):
    cursor.execute(f"SELECT rowid FROM {table} WHERE {column} = ?", (value,))
    record = cursor.fetchone()
    if record:
        return record[0]
    cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
    return cursor.lastrowid


# In[193]:


for idx, row in data.iterrows():
    platform_id = get_or_create(c, 'platforms', 'platform', row['Platform'])
    genre_id = get_or_create(c, 'genres', 'genre', row['Genre'])
    publisher_id = get_or_create(c, 'publishers', 'publisher', row['Publisher'])

    # Insert game details
    c.execute("INSERT INTO games (genre_id, name, year) VALUES (?, ?, ?)", (genre_id, row['Name'], row['Year']))
    game_id = c.lastrowid

    # Insert sales data
    c.execute("""
    INSERT INTO sales (game_id, platform_id, publisher_id, na_sales, eu_sales, jp_sales, other_sales, global_sales)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (game_id, platform_id, publisher_id,
                                         row['NA_Sales'], row['EU_Sales'],
                                         row['JP_Sales'], row['Other_Sales'], 
                                         row['Global_Sales']))


# In[195]:


pd.read_sql("""
SELECT *
FROM platforms
""", conn)


# In[196]:


pd.read_sql("""
SELECT *
FROM sales
""", conn)


# In[179]:


conn.commit()
conn.close()


# In[ ]:





# In[ ]:




