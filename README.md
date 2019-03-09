# Automated Essay Grading using Latent Semantic Analysis

## Abstract

Accurate and consistent Automated Essay Grading (AES) has been a challenge of machine learning and cognitive psychology, 
with interest increasing as a result of the increased usage of standardized testing in education (Shermis, 2014). 
In a meta-review on the 2012 Kaggle AES competition, Natural Language Processing through Recurrent Neural Networks, 
statistical analysis using Bayesian Models and Latent Semantic Analysis (LSA) were the most promising solutions 
implemented (Shermis, 2014). In this paper, we implement Generalized LSA (GLSA) on the Kaggle dataset, 
and investigate the ways that varying dimensions and n-gram ranges influence the accuracy of the model. 
Our implementation achieved accuracy high enough to support the idea that some semantic meaning of words 
can be based on contextual similarity. However, the shortcomings of the model suggest that context-based similarities 
of words cannot fully capture the complexities of evaluating essays. We found that the optimal n-grams and dimensions 
are not consistent across the 8 essay questions in the dataset, yet they reveal trends that provide insight into the 
underlying function of GLSA.

## Keywords
Automated Essay Grading; Latent Semantic Analysis; Singular Vector Decomposition; N-Grams.


## References

* Educational Testing Service. (2019). ETS Research: Automated Scoring of Writing Quality. Retrieved from https://www.ets.org/research/topics/as_nlp/writing_quality
* Islam, M. M., & Hoque, A. L. (2010). Automated essay scoring using generalized latent semantic analysis. In 2010 13th International Conference on Computer and Information Technology (ICCIT) (pp. 358–363). IEEE.
* Kintsch, W. (2002). The potential of latent semantic analysis for machine grading of clinical case summaries. Journal of Biomedical Informatics, 35(1), 3–7.
* Landauer, T. K., Foltz, P. W., & Laham, D. (1998). An introduction to latent semantic analysis. Discourse Processes, 25(2–3), 259–284.
* Shermis, M. D. (2014). State-of-the-art automated essay scoring: Competition, results, and future directions from a United States demonstration. Assessing Writing, 20, 53–76.
* The Hewlett Foundation. (2012). Overview. Retrieved from https://www.kaggle.com/c/asap-aes

## Paper
[Final Paper](AES_paper.pdf)

## To run the program

### Python dependencies:

* Sklearn
* Scipy
* Numpy
* Matplotlib
* Nltk
* Seaborn

To install:

```python
pip3 install sklearn scipy numpy matplotlib nltk seaborn
```
### To run:

1. Import Classes

```python
Import preprocessor
Import run
Import visualizer
```

2.	Run “preprocessor.main()” to process the all essays across all 8 questions:
```python
preprocessor.main()
```
*	Removes stopwords, lemmatizes and tokenizes essays (using clean_and_tokenize)
*	Scales the scores to be consistent across all questions to 1 and 10 and splitting them into train, validation and test set and then saving the 6 datasets - essay_train_set, essay_valid_set, essay_test_set, score_train_set, score_valid_set, score_test_set – into dataset.npz in the “data” Folder(using run)

3.	Run “run.main()” to get the optimal ngram-dimension pair and the corresponding F1 scores in the validation and test set across all 8 questions and ngram ranges
*The program ran on GCP compute engines and AWS EC2 with 96 vCPUs and only completed in 2 nights*
```python
run.main()
```
*	For every question, iterates through 18 ngram ranges to build ngram-by-documents to apply SVD to (using vectorizer).
*	Truncate the SVD components in 30 dimensions and use them to reduce the training and validation set. Assign the scores and calculate the F1 scores on the validation set. For every ngram range the optimal dimension is the one which gives the greatest F1 score on the validation set (using metrics).
*	Then try the 18 ngram with optimal dimensions on the test set to get new F1 scores. The ngram-dimension pair which gives the greatest F1 score on the test set is the optimal ngram and dimension to use in a question. 
*	Iterate across all 8 questions and store the validation and test statistics in a gzipped csv table in the “data” Folder

4.	Run “visualizer.main()” to get all the graphs in the “img” Folder
```python
visualize.main()
```

#### Folder directory

```
|____metrics.py (all the python run code)
|____run.py
|____preprocessor.py
|____vectorizer.py
|____visualize.py

|____training_set.csv (the essay and score dataset we use)

|____img (directory containing the matplotlib graphs)
|____data (directory containing the processed essay data and statistics)
| |____qn 1-5 (For question 1 to 5, directory containing validation and test results by ngram)
| |____qn 6-8 (For question 6 to 8, directory containing validation and test results by ngram)
| |____dataset.npz (For all questions, dataset containing the train, validation and test essays and scores)
```