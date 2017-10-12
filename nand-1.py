
# coding: utf-8

# In[6]:

import numpy as np
class Perceptron:
    
    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.ones(input_length) * 0.5
        else:
            self.weights = weights
        
    @staticmethod
    def unit_step_function(x):
        if x > 0.5:
            return 1
        return 0
        
    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)
    
p = Perceptron(2, np.array([0.5, 0.5]))
for x in [np.array([0, 0]), np.array([0, 1]), 
          np.array([1, 0]), np.array([1, 1])]:
    y = p(np.array(x))
    print(x, y)


if __name__ == "__main__":
	# input data
	in_a = np.array([0,0,1,1,0])
	in_b = np.array([0,1,0,1,0])

	# output ground truth (OR gate)
	o_gt = np.array([1,1,1,0,1])

	# weights
	w = np.random.randn(1,2)
	bias = np.random.randn(1,1)

	print (format(w))
	print (format(bias))

	
	learning_rate = 1
	epochs = 10

	
	# TRAINING
	x_axis = []
	y_axis = []
	for e in range(epochs):
		error = 0
		for a, b, o in zip(in_a, in_b, o_gt):
			# weighted sum
			ws = w[0][0]*a + w[0][1]*b
			# activation
			summed = ws + bias
			# initialize output to zero
			y = 0
			if summed >= 0:
				y = 1
			else:
				y = 0
			
			# calculate the error
			error = o - y
			
			# update rule
			dw0 = learning_rate * (error) * a
			dw1 = learning_rate * (error) * b

			# update weights
			w[0][0] = w[0][0] + dw0
			w[0][1] = w[0][1] + dw1

			# update bias too!
			bias = bias + (learning_rate * error)
			
			# for visualization purpose
			x_axis.append(e)
			y_axis.append(error)

	print (format(w))
	print (format(bias))

	# TESTING
	perceptron_output = []
	for a, b in zip(in_a, in_b):
		# weighted sum
			ws = w[0][0]*a + w[0][1]*b
			# activation
			summed = ws + bias
			# initialize output to zero
			if summed >= 0:
				perceptron_output.append(1)
			else:
				perceptron_output.append(0)

	# summary
	print (format(in_a))
	print (format(in_b))
	print (format(o_gt))
	print (format(perceptron_output))


# In[ ]:



