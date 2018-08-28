import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(sum(map(ord, "categorical")))


x_data = [50, 100, 150, 200, 250, 300]
y_data = [13766, 15360, 15114, 16405, 18085, 21418]
y_data = [i /60 for i in y_data]

df = pd.DataFrame({"Time (mins)": y_data, "Image Size (n x n)": x_data})


titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# sns_plot = sns.barplot(x="Size", data=df, palette="Greens_d")
sns_plot = sns.barplot(x="Image Size (n x n)", y="Time (mins)", palette="Greens_d", data=df, dodge=False).set_title('Image Size Effect on Training Time')
fig = sns_plot.get_figure()
fig.savefig('output.eps', format='eps', dpi=1200)

plt.show()