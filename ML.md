->**Label**:The thing we are predicting (y)
->**Feature**: The data we are evaluating from an example (x)
->**Example**:A single instance of the thing which we are taking features.
  - Labeled
  - Unlabeled
**Batch:** Total number of examples you use to calculate the gradient in a   single iteration.
**Hyperparameter:** Parameter whose value is set before the learning    process begins and can directly affect how well a model trains.
- [Learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)
-   Number of [epochs](https://developers.google.com/machine-learning/glossary/#epoch)
-   [Batch size](https://developers.google.com/machine-learning/glossary/#batch_size)


====**Linear Relationship (y = mx + b)

In ML, it is: y' = b + w1*x1, if we had more features, y' = b + w1*x1 + w2*x2...

y' = label
x = feature
w = slope of the line = weight of feat X1 (How much the y' changes for a unit change in x1)
b = bias term = y-intercept (where the line croses y-axis)

--

Training a model simply means learning (determining) good values for all the weights
and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.


**Squared Loss aka L2 loss (Loss function)**

loss = (observation - prediction(x))² = (y - y')²

**Mean square error (MSE)** is the average squared loss per example over the whole dataset.
![[Pasted image 20230316231151.png]]

====**Reducing loss iteratively**

The "model" takes one or more features as input and returns one prediction (y') as output.
So, we establish initial values for w1 and b, for example w1 = 0; b = 0. These values are not important (can be random as well). 
The "Compute Loss" part comes up examinating the ouput and generating new values for w1 and b, doing this iteratively, we sometime will reach "low loss" values for b and w1.

**Gradient Descent**
Suppose we had the time and the computing resources to calculate the loss for all possible values of w1, this would be the chart.
![[Pasted image 20230316231815.png]]
Doing this is very inefficient, a better mechanism is **gradient descent:**

1. Take random values for w1 (start)
![[Pasted image 20230316232154.png]]
2.**Gradient** descent algorithm then calculates the gradient of the loss curve at the starting point.

-->**Gradient** is a vector, so it has both of the following characteristics:
-   A direction
-   A magnitude

![[Pasted image 20230316232455.png]]

3.The gradient descent algorithm takes a step in the **direction** of the negative gradient in order to reduce loss as quickly as possible. To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's **magnitude** to the starting point as shown in the following figure.

   The gradient descent then repeats this process, edging ever closer to the minimum.


**Learning rate

Gradient vector has direction and magnitude, gradient descent algorithm calculates the next point multiplying the gradient by a scalar (**Learning rate** or step size).

For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

**Hyperparameters** are the knobs that programmers tweak in machine learning algorithms (like the learning rate).

![[Pasted image 20230316233238.png]]
![[Pasted image 20230316233256.png]]
![[Pasted image 20230316233308.png]]
		The ideal learning rate in one-dimension is 1/f(x)″ (the inverse of the second derivative of f(x) at x).

**Stochastic gradient descent**

*A large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.*

**What if we could get the right gradient on average for much less computation?**
By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one.

->**Stochastic gradient descent** (**SGD**) takes this idea to the extreme it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.
->**Mini-batch stochastic gradient descent** (**mini-batch SGD**) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------
![[Pasted image 20230317000423.png]]
Note the following about the model visualization:

-   Each axis represents a specific feature. In the case of spam vs. not spam, the features could be the word count and the number of recipients of the email.
    
    **Note:** Appropriate axis values will depend on feature data. The axis values shown above would not make sense for word count or number of recipients, as neither can be negative.
    
-   Each dot plots the feature values for one example of the data, such as an email.
-   The color of the dot represents the class that the example belongs to. For example, the blue dots can represent non-spam emails while the orange dots can represent spam emails.
-   The background color represents the model's prediction of where examples of that color should be found. A blue background around a blue dot means that the model is correctly predicting that example. Conversely, an orange background around a blue dot means that the model is incorrectly predicting that example.
-   The background blues and oranges are scaled. For example, the left side of the visualization is solid blue but gradually fades to white in the center of the visualization. You can think of the color strength as suggesting the model's confidence in its guess. So solid blue means that the model is very confident about its guess and light blue means that the model is less confident. (The model visualization shown in the figure is doing a poor job of prediction.)

Use the visualization to judge your model's progress. ("Excellent—most of the blue dots have a blue background" or "Oh no! The blue dots have an orange background.") Beyond the colors, Playground also displays the model's current loss numerically. ("Oh no! Loss is going up instead of down.")

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


===**Introduction to TensorFlow and AI programming**

**NUMPY**

```py
import numpy as np

one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])

two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
sequence_of_integers = np.arange(5, 12)

random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6,3))

random_floats_between_0_and_1 = np.random.random([6])

# Sums al elements one by one
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0 

#multiplies Eij * 3  one by 1 
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3 
```

**PANDAS**


```py
import pandas as pd

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

#Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

#Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)


#Subsets of the dataframe

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')
print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')
print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')
print("Column 'temperature':")
print(my_dataframe['temperature'])
```


**TFKERAS**

->Import TF.

```py
import tensorflow as tf
```

->The dataset consists of 12 examples. Each example consists of one feature and one label.

```py
The dataset consists of 12 examples. Each example consists of one feature and one label.
my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
```

->The following code cell initializes these hyperparameters and then invokes the functions that build and train the model.

```py
learning_rate=0.01
epochs=10
my_batch_size=12


my_model = build_model(learning_rate)

trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, my_label, epochs,
my_batch_size)

plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

Executing this will result in this two graphs: 

*Model
![[Pasted image 20230317022253.png]]
Loss*
![[Pasted image 20230317022300.png]]
In the model graph, the red line is the output of the trained model, and it should align as much as possible with the blue dots.
About the loss one, the fact of it doesn´t flatten out tells us that the model hasn´t trained sufficiently. Solution:

*Increase number of epochs*

```py 
learning_rate=0.01
epochs= 200 
my_batch_size=12

my_model = build_model(learning_rate)

trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, my_label, epochs,
my_batch_size)

plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

result:

*Model*
![[Pasted image 20230317023106.png]]
*Evolution trough epochs*
![[Pasted image 20230317023157.png]]

**BAD CHOSEN LEARNING RATE (Too high)

![[Pasted image 20230317023421.png]]

**BEST CHOICE 

![[Pasted image 20230317023611.png]]

***Adjust the batch size**

->System recalculates the model's loss value and adjusts the model's weights and bias after each iteration.

->==**Each iteration**== is the span in which the system processes ==**one batch**==.
* 1iteration ----> 1batch

If the batch size is 6, then the system recalculates the model's loss value and adjusts the model's weights and bias after processing every 6 examples.

->One **epoch** spans sufficient iterations to process every example in the dataset.
* 1epoch ----> entireDataSet

If the batch size is 12 (dataset size is 12), then each epoch lasts one iteration. However, if the batch size is 6, then each epoch consumes two iterations.

*""It is tempting to simply set the batch size to the number of examples in the dataset (12, in this case). However, the model might actually train faster on smaller batches. Conversely, very small batches might not contain enough information to help the model converge.""*

```
That said, here are a few rules of thumb for hyperparameter tuning:

->   Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
->  If the training loss does not converge, train for more epochs.
->  If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
->  If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
->  Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
->  Setting the batch size to a _very_ small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
->  For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.
```