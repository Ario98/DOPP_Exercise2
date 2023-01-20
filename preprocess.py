import pandas as pd

# Read data
df = pd.read_csv("data/games.csv")

# Drop rows containing Playtest in the name as not needed
df['Name'] = df['Name'].astype(str)
df = df[df["Name"].str.contains("playtest|demo")==False]

# Drop columns containing too much empty data
df = df.drop(['Reviews','Website','Support url','Support email','Metacritic url',
    'Metacritic score','User score','Score rank','Notes', 'About the game', 'Header image', 'AppID', 'Name', 'Screenshots', 'Movies', 'Release date'
    , 'Supported languages', 'Full audio languages', 'Developers', 'Publishers'], axis=1) # this row for discussion

# Fill NA values in string columns
df['Tags'] = df['Tags'].fillna("None")
#df['Developers'].fillna("Unknown", inplace=True)
#df['Publishers'].fillna("Unknown", inplace=True)
df = df[df['Categories'].notna()]

# Subset dataframe based on if column contains NaN values or if the game is a playtest
df = df[df['Genres'].notna()]
df = df[df['Categories'].notna()]

# Drop rows containing 0 in both Positive and Negative columns
df = df[~(df['Positive'] == 0) & (df['Negative'] == 0)]

# Expand the columns containing genres, tags and categories. Also languages
df = df.join(df.Categories.str.get_dummies(',').add_prefix("category_"))
df = df.join(df.Genres.str.get_dummies(',').add_prefix("genre_"))
df = df.join(df.Tags.str.get_dummies(',').add_prefix("tag_"))
#df = df.join(df['Supported languages'].str.get_dummies(',').add_prefix("lang_"))
#df = df.join(df['Full audio languages'].str.get_dummies(',').add_prefix("audio_lang_"))

# Encoding the target variable using the midpoint
# Create a new column with the midpoint of the range
df['Estimated Owners Midpoint'] = df['Estimated owners'].apply(lambda x: (int(x.split(' - ')[0]) + int(x.split(' - ')[1]))/2)
df.drop(['Estimated owners'], axis=1, inplace=True)

# Encode TRUE and FALSE columns
df = df.replace({True: 1, False: 0})

# Drop exploded columns
df = df.drop(['Categories', 'Genres', 'Tags'], axis=1)

df.to_excel("data/preprocessed.xlsx", index=False)