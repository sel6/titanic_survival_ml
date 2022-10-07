import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

class Data_Visualisation():

    def plot_bargraph(data, x):
        """
        Plot bargraph
        """
        plt.figure(figsize=(5, 5))
        sns.countplot(x=x, data=data, palette='Accent_r')
        return(pd.value_counts(data[x]))
    
    def pair_barplot(column1, column2, data):
        """
        Plot pair bargraph
        """
        fig, (axis1, axis2) = plt.subplots(1,2,figsize=(24,24))
        sns.countplot(x=column1, data=data, palette='Accent_r', ax=axis1)
        sns.countplot(x=column2, data=data, palette='Accent_r', ax=axis2)
        return(pd.value_counts(data[column1]), " ", pd.value_counts(data[column2]))

    def get_list_difference1(lis1, lis2):
        """
        A function to get list difference if one of the list element is negative
        """
        arr1 = np.array(lis1)
        arr2 = np.array(lis2)
        subtracted_array = np.add(arr1, arr2)
        subtracted = list(subtracted_array)

        return(subtracted)

    def get_list_difference2(lis1, lis2):
        """
        A function to get list difference if both of list elements are positive
        """
        arr1 = np.array(lis1)
        arr2 = np.array(lis2)
        subtracted_array = np.subtract(arr1, arr2)
        subtracted = list(subtracted_array)

        return(subtracted)

    def plot_difference(lis1, lis2, lis3, label_lis, leg_lis):
        """
        A function to plot the differnce between two graphs
        """

        sns.set_style("darkgrid")

        plt.xlabel(label_lis[0])
        plt.ylabel(label_lis[1])
        
        
        plt.plot(lis3, lis1, marker = 'x')
        plt.plot(lis3, lis2, marker = '.')
        
        plt.title("Differnce Graph")
        plt.legend(leg_lis)
        
       

    def show_distribution(df, col, color, title):
        """
        A function to plot distrbuition plot
        """
        sns.displot(data=df, x=col, color=color, kde=True, height=4, aspect=2)
        plt.title(title, size=15, fontweight='bold')
        plt.show()
        
    def swarm_plot(df, column):
        """
        A function to plot swarm plot
        """
        sns.set(rc={'figure.figsize':(8,8)})
        sns.swarmplot(x=df[column])  
    
    def plot_heatmap(df, title):
        """
        A function to plot heatmap
        """
        f, ax = plt.subplots(figsize = ( 7, 7))
        ax = plt.axes()
        sns.set(font_scale=.8)
        sns.heatmap(df.corr(), annot = True, linewidth = 0.5, fmt = '.1f', ax = ax )
        ax.set_title(title)
        

    def plot_pie(lis1, lis2, col, title1, title2):
        """
        A function to plot a pie chart
        """

        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%".format(pct)

        # Creating plot
        fig, ax = plt.subplots(figsize =(4, 4))
        wedges, texts, autotexts = ax.pie(lis2,
                                          autopct = lambda pct: func(pct, lis2),
                                          labels = lis1,
                                          shadow = False,
                                          # colors = colors,
                                          startangle = 190,
                                          textprops = dict(color ="magenta"))

        # Adding legend
        ax.legend(wedges, lis1,
                  title =title1,
                  loc ="center left",
                  bbox_to_anchor =(1, 0, 0.5, 1))
        
        for autotext in autotexts:
            autotext.set_color('black')
        plt.setp(autotexts, size = 13)
        ax.set_title(title2)

        # show plot
        plt.show()
        return(pd.value_counts(col))
    
    def joint_plot(df, col1, col2):
        """
        A function to plot joint plot
        """
        sns.set(style="white", color_codes=True)
        jp=sns.jointplot(df.loc[:,col1], df.loc[:,col2], kind="reg",color="b")
        r, p = stats.pearsonr(df.loc[:,col1], df.loc[:,col2])
        jp.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',
                        xy=(0.1, 0.9), xycoords='axes fraction',
                        ha='left', va='center',
                        bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})