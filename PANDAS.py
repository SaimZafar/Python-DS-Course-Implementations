import pandas as pd
pd.__version__
df=pd.read_csv(r'C:\Users\hp\Downloads\california_housing_small.csv')
print(df)
print(df.head(3))  # head displays starting nth rows
print(df.tail(4))  # tail displays ending nth rows
print(df.columns)  # tell names of columns

# creating new column
df["houseage_per_avgrooms"]=df["HouseAge"]/df["AveRooms"]
print(df)

# renaming columns
df=df.rename(columns={"AveRooms":"AvgRms","HouseAge":"HsAge"})
print(df)

# extracting columns from one df
df2=df[["MedInc","HsAge","AvgRms"]]
print(df2)
print(df2.head())

# selecting rows
print(df.iloc[4:10])
print(df[df["HsAge"]>30]) # displays rows having hsage >30 using condition

# extraction of both rows and columns
print(df.iloc[3:9,1:3])

# condition statement and column selection
dfe=df.loc[df["HsAge"]>30,"MedInc"]
print(dfe)


# AGGREGATING STATISTICS
#########################
#########################

#calculating mean and median
print()
print("mean of house age is: ",df["HsAge"].mean())
print("median of house age is: ",df["HsAge"].median())

# description of the whole data set or selected ones
print("description of the data set is:\n",df.describe())
print("description of the  selected data set: \n",df[["HsAge","MedInc"]].describe())

# column category frequency
print(df["HsAge"].value_counts())
















