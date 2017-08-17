Question 2 - Model Application

List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen

    Describe one real-world application in industry where the model can be applied.
    What are the strengths of the model; when does it perform well?
    What are the weaknesses of the model; when does it perform poorly?
    What makes this model a good candidate for the problem, given what you know about the data?

HINT:

Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

Answer:

* Support Vector Machines (LinearSVC)
* Decision Trees (DecisionTreeClassifier)
* K-Nearest Neighbors (KNeighborsClassifier)



Question 3 - Choosing the Best Model

    Based on the evaluation you performed earlier, in one to two paragraphs, explain to CharityML which of the three models you believe to be most appropriate for the task of identifying individuals that make more than $50,000.

HINT: Look at the graph at the bottom left from the cell above(the visualization created by vs.evaluate(results, accuracy, fscore)) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:

    metrics - F score on the testing when 100% of the training data is used,
    prediction/training time
    the algorithm's suitability for the data.

Answer: LinearSVC


Question 4 - Describing the Model in Layman's Terms

    In one to two paragraphs, explain to CharityML, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.

HINT:

When explaining your model, if using external resources please include all citations.

Answer:

Question 5 - Final Model Evaluation

    What is your optimized model's accuracy and F-score on the testing data?
    Are these scores better or worse than the unoptimized model?
    How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in Question 1?_

Note: Fill in the table below with your results, and then provide discussion in the Answer box.
Results:
Metric 	        Benchmark Predictor 	Unoptimized Model 	Optimized Model
Accuracy Score
F-score 			EXAMPLE

Answer:
