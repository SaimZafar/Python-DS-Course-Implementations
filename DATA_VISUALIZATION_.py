import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.__version__
white=pd.read_csv(r'C:\Users\hp\Downloads\whitewine_sample.csv')
print(white)

# histogram using matplotlib
fig, ax = plt.subplots()
ax.hist(white['quality'])
ax.set_title('Wine Review Scores')
ax.set_xlabel('pH')
ax.set_ylabel('Frequency')
plt.show()

# histogram using pandas
white['quality'].plot.hist()
plt.show()

#density plot using pandas
white['quality'].plot(kind="density")
plt.show()

#density plot using seaborn
sns.kdeplot(white['quality'],fill=True)
plt.show()

#box plot using seaborn
sns.boxplot(x="quality",data=white)
plt.show()

#bar plot using seaborn
sns.countplot(x="quality",data=white)
plt.show()

#scatter plot using seaborn: CATEGORICAL VS NUMERICAL
sns.scatterplot(x="alcohol",y="pH",data=white)
plt.show()

#scatter plot using seaborn: NUMERICAL VS NUMERICAL
sns.scatterplot(x="quality",y="pH",data=white)
plt.show()







