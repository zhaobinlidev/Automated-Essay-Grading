import glob
import matplotlib as mpl
mpl.rc('savefig', format='eps', bbox='tight')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("paper")
sns.set_palette('dark')

class Visualizer:
    def __init__(self):
        pass

    def run(self, results_dir):
        '''
        Run “visualizer.main()” to get all the graphs in the “img” Folder
        :param results_dir: directory containing the results
        :return: generate matplotlib graphs in "img" Folder and return the optimum_results_df
        '''
        metrics_valid_df, metrics_test_df = self.get_results(results_dir)
        optimum_results_df = self.get_optimum_results(metrics_test_df)

        self.plt_F1_score_against_dimension(metrics_valid_df)
        self.plt_ngram_by_dimension_question_1(metrics_valid_df)
        self.plt_optimum_dimension_by_ngram(metrics_test_df)
        self.plt_optimum_dimensions_by_question(optimum_results_df)
        self.plt_F1_score_by_ngram(metrics_test_df)
        self.plt_optimum_F1_score_by_question(optimum_results_df)
        
        return metrics_valid_df, metrics_test_df, optimum_results_df
    
    def get_results(self, results_dir):
        metrics_valid_list = []
        metrics_test_list = []

        for path in glob.iglob(results_dir + '/**/*.csv*', recursive=True):
            if 'metrics' in path:
                if 'valid' in path:
                    metrics_valid_list.append(pd.read_csv(path))
                elif 'test' in path:
                    metrics_test_list.append(pd.read_csv(path))

        metrics_valid_df = pd.concat([df for df in metrics_valid_list], ignore_index=True)
        metrics_test_df = pd.concat([df for df in metrics_test_list], ignore_index=True)

        return metrics_valid_df, metrics_test_df

    def get_optimum_results(self, metrics_test_df):
        idx = metrics_test_df.groupby(['question_no'])['score_F1'] \
                  .transform(max) == metrics_test_df['score_F1']

        optimum_results_df = metrics_test_df[idx].sort_values(['question_no']).loc[:,
                             metrics_test_df.columns != 'max_sim_scores']

        with open('optimized_results.tex', 'w') as f:
            optimized_latex = optimum_results_df.rename(
                {"question_no": 'Qn', "optimum_ngram_range": "Ngram", "optimum_dimensions": "Dims",
                 "score_F1": "F1", "max_sim_acc": "% correct", "max_sim_err": "Abs. Error"}, axis=1). \
                round(3).to_latex(index=False, bold_rows=True)
            f.write(optimized_latex)

        return optimum_results_df
  
        
    def plt_F1_score_against_dimension(self, metrics_valid_df):
        #Figure 4: F1 Score Across Dimensionalities for Question 1, n-gram (5,15)
        
        ngram = '(5, 15)'
        question = 1
        metrics_valid_plt = metrics_valid_df.loc[
            (metrics_valid_df['question_no'] == question) & (metrics_valid_df['ngram_range'] == ngram)]
        max_row = metrics_valid_plt.loc[metrics_valid_plt['score_F1'].idxmax()]
        max_dim, max_score = max_row['dimensions'], max_row['score_F1']
        
        ax = metrics_valid_plt.plot(x="dimensions", y='score_F1', legend=False)
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('F1 score')
        ax.set_title(
            'Question ' + str(question) + ', n-gram ' + ngram + ': ' + 'F1 score by dimensions')
        
        ax.plot(max_dim, max_score, 'o')
        ax.annotate('optimal dimension', xy=(max_dim, max_score), xytext=(500, 0.2575), arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.savefig(
            fname='img/F1_score_against_dimension_score_F1_question_' + str(question) + '.eps')
        plt.show()
        plt.clf()
    
    def plt_ngram_by_dimension_question_1(self, metrics_valid_df):
        #Figure 5
        question = 1
        metrics_valid_plt = metrics_valid_df.loc[(metrics_valid_df['question_no'] == question)]

        ax = sns.lineplot(data=metrics_valid_plt, x='dimensions', y='score_F1', hue="ngram_range")
        ax.set_xlabel('Dimensions')
        ax.set_ylabel("F1 score")
        ax.set_title('Question ' + str(question) + ': F1 score by dimensions per ngram')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig('img/ngram_by_dimension_per_question_score_F1.eps')
        plt.show()
        plt.clf()

    def plt_ngram_by_dimension_per_question(self, metrics_valid_df):
        #Like Figure 5 and including every question
        
        g = sns.FacetGrid(data=metrics_valid_df, col="question_no", hue="ngram_range", col_wrap=3)
        g = (g.map_dataframe(plt.plot, "dimensions", "score_F1")).add_legend().set_axis_labels("Dimensions", "F1 Score").set_titles("Question: {col_name}")
        g.savefig('img/ngram_by_dimension_per_question_score_F1.eps')
        plt.show()
        plt.clf()
    
    def plt_optimum_dimension_by_ngram(self, metrics_test_df):
        #Figure 6: Optimum Dimensionalities for all n-gram ranges in Question 1

        question = 1
        metrics_test_plt = metrics_test_df.loc[(metrics_test_df['question_no'] == question)].sort_values(
            by="optimum_dimensions",
            ascending=False)
        ax = metrics_test_plt.plot(x="optimum_ngram_range", y="optimum_dimensions", kind='bar', legend=False)
        ax.set_xticklabels([ngram for ngram in metrics_test_plt["optimum_ngram_range"].values])
        ax.set_xlabel('n-gram Range')
        ax.set_ylabel("Optimal Dimension")
        ax.set_title('Question ' + str(question) + ': Optimal Dimension by n-gram Range')
        plt.savefig(
            fname='img/optimum_dimensions_by_ngram_qn_' + str(question) + '.eps')
        plt.show()
        plt.clf()
    
    def plt_optimum_dimensions_by_question(self, optimum_results_df):
        #Figure 7: Optimal Dimensions by Question

        ax = optimum_results_df.plot(x="question_no", y="optimum_dimensions", kind='bar', legend=False)
        ax.set_xlabel('Question no')
        ax.set_ylabel("Optimum Dimensions")
        ax.set_title('Optimum Dimensions by Questions')
        plt.savefig(
            fname='img/optimum_dimensions_by_question.eps')
        plt.show()
        plt.clf()
        
    def plt_F1_score_by_ngram(self, metrics_test_df):
        #Figure 8: Question 1 F1 Score for each n-gram
        
        question = 1
        metrics_test_plt = metrics_test_df.loc[(metrics_test_df['question_no'] == question)].sort_values(by="score_F1",
                                                                                                  ascending=False)
        ax = metrics_test_plt.plot(x="optimum_ngram_range", y="score_F1", kind='bar', legend=False)
        ax.set_xticklabels([ngram for ngram in metrics_test_plt["optimum_ngram_range"].values])
        ax.set_xlabel('n-gram Range')
        ax.set_ylabel("F1 score")
        ax.set_title('Question ' + str(question) + ': F1 score by n-gram Range')
        plt.savefig(
            fname='img/F1_score_by_ngram_qn_' + str(question) + '.eps')
        plt.show()
        plt.clf()
       
    def plt_optimum_F1_score_by_question(self, optimum_results_df):
        #Figure 9: Model F1 Score by Question
        
        ax = optimum_results_df.plot(x="question_no", y="score_F1", legend=False, kind='bar')
        ax.set_xticklabels(
            [str(question) + '\n' + str(ngram) for (question, ngram) in 
             zip(optimum_results_df["question_no"].values, optimum_results_df["optimum_ngram_range"].values)], rotation=30)
        ax.set_xlabel('Question no and Optimum ngram')
        ax.set_ylabel("F1 score")
        ax.set_title('F1 score by Questions')
        plt.savefig(
            fname='img/optimum_F1_score_by_question.eps')
        plt.show()
        plt.clf()
        
def main():
    visualizer = Visualizer()
    visualizer.run("data")


if __name__ == '__main__':
    main()
    