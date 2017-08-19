Question 2 - Model Application

List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen

    Describe one real-world application in industry where the model can be applied.
    What are the strengths of the model; when does it perform well?
    What are the weaknesses of the model; when does it perform poorly?
    What makes this model a good candidate for the problem, given what you know about the data?

HINT:

Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

Answer:

### Support Vector Machines (LinearSVC)

Example Application: Computational Biology.
Strengths: Effective in high-dimensional spaces, memory efficient, and fast(Linear SVC).
Weaknesses: Does not provide probability estimates.
Applicability: The census data has a large number of features(especially after on-hot encoding), and will potentially be used on large data sets so will benefit from an efficient algorithm.

^ http://scikit-learn.org/stable/modules/svm.html
^ https://noble.gs.washington.edu/papers/noble_support.html

### Decision Trees (DecisionTreeClassifier)

Example Application:: Medical Diagnosis.
Strengths: Simple to understand, Fast prediction, Can handle numerical and categorical data directly.
Weaknesses: Tendency for overfitting, diffculty to learn XOR/parity based concepts, dataset needs to be balanced to avoid bias.
Applicability: The census data has a mixed data, and will potentially be used on large data sets so will benefit from fast prediction. Unlikely to contain parity type relationships that affect the income class labels.

^ http://scikit-learn.org/stable/modules/tree.html
^ http://www.cbcb.umd.edu/~salzberg/docs/murthy_thesis/survey/node32.html

### K-Nearest Neighbors (KNeighborsClassifier)

Example Application:: Visual Pattern Recognition.
Strengths: Simplicity, Fast training, Flexible Algorithms.
Weaknesses: Prediction cost/time can be high, Lazy learner, Can have problems when there a many features(curse of dimensionality).
Applicability: The fast training time will be a benefit on large census data sets.

^ http://scikit-learn.org/stable/modules/neighbors.html
^ http://www.cs.cornell.edu/courses/cs472/2005fa/lectures/7-knn_6up.pdf

Question 3 - Choosing the Best Model

    Based on the evaluation you performed earlier, in one to two paragraphs, explain to CharityML which of the three models you believe to be most appropriate for the task of identifying individuals that make more than $50,000.

HINT: Look at the graph at the bottom left from the cell above(the visualization created by vs.evaluate(results, accuracy, fscore)) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:

    metrics - F score on the testing when 100% of the training data is used,
    prediction/training time
    the algorithm's suitability for the data.

Answer: LinearSVC

    Out of the three models used I would recommend using LinearSVC based on these tests. The F-score of 0.685 and accuracy of 0.843 on the unoptimised model is clearly the best of the three options. In addition although training time was the the slowest of the three, prediction time was fast, as was K-NN.

    Looking at the accuracy scores between training/testing sets both K-NN and decision tree show signs of overfitting as their training scores trend higher than testing scores whereas in this case SVM improves on testing scores as the training set size increases.


Question 4 - Describing the Model in Layman's Terms

    In one to two paragraphs, explain to CharityML, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.

HINT:

When explaining your model, if using external resources please include all citations.

Answer:

LinearSVC attempts to classify data points by separating them by a line(or plane) that gives the best margin for error in that the separating plane is as far away as possible from the points on each side of it.

 Training the model involves identifying which points have the most importance(or weight) in defining the separating plane - which in general will be the points closest to it. Once the model is trained prediction can be quite fast as the query point is classified by plotting its position relative to the separating plane that was found in the training phase.


Question 5 - Final Model Evaluation

    What is your optimized model's accuracy and F-score on the testing data?
    Are these scores better or worse than the unoptimized model?
    How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in Question 1?_

Note: Fill in the table below with your results, and then provide discussion in the Answer box.
Results:
Metric 	        Benchmark Predictor 	Unoptimized Model 	Optimized Model
Accuracy Score        0.2478                 0.8427              0.8430
F-score 			  0.2917                 0.6856              0.6874

Answer:

The optmised scores are slightly better that the un-optimised model in that the accuracy is higher and also precision vs recall (by the f0.5-score) is a little higher.

Compared to the naive predictor benchmarks the accuracy is improved from 24% to 84%. And the F(0.5)-score from 0.29 to 0.69. This could be considered a very good result compared to the naive benchmark.

Question 6 - Feature Relevance Observation

When Exploring the Data, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

Answer:

The following outlines the order I chose based on intuition:

__age__ - In general more experience and seniority will lead to more income also more time to accumulate income earning assets.
__hours-per-week__ - More hours worked should lead to higher overall pay than less hours.
__occupation__ - Some occupations are more highly paid than others.
__education__ - Higher education level may indicate more income earning ability, depending on occupation.
__capital-gain__ - Capital gain may be an indicator of investment activity which could indicate income level.


Question 7 - Extracting Feature Importance

Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above $50,000.

    How do these five features compare to the five features you discussed in Question 6?
    If you were close to the same answer, how does this visualization confirm your thoughts?
    If you were not close, why do you think these features are more relevant?

Answer:

    Capital loss/ Capital Gain - I did not expect this, but then investment activity may lead to loss/or gain so this make sense. Higher income earners may also have tax-incentives to invest.
    Age - This was first on my list so this helps confirm my intuition there.
    hours-per-week - Also featured highly as expected, just not as high as capital gain/loss
    eduction-num - Total years of education. This is interesting as seems to have higher weight than the actual level acheived.

    NB. One factor which I think skews this weighting is that features shown are the numerical ones. Given that the categorical features were one-hot encoded this would spread out their individual predictive influence.

Question 8 - Effects of Feature Selection

    How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
    If training time was a factor, would you consider using the reduced data as your training set?


Answer:

    In this case the accuracy and f-score are reduced although not drastically considering only 5 features of the original. If training time was a factor I would definitly consider it for SVM as the training time was reduced significantly when using only the 5 features.
