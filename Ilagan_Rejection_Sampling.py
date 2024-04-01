#!/usr/bin/env python
# coding: utf-8

# *Carlito James Ilagan* | *B37* | *Date Submitted*

# <h2>No. 1</h2>

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# the function
def distribution(x):
    return np.exp(x**2) * (np.cos(2*x)**2 * np.sin(5*x) + np.sin(x)**2 + 1)
# Uniform distribution
def Uniform_distribution(lower, upper, size):
    return np.random.uniform(lower, upper, size)
# the rejection sampling code
def rejection_sampling(samples):
    accepted_samples = []
    rejected_samples = []

    while len(accepted_samples) < samples:
        x = Uniform_distribution(a, b, 1)[0]
        y = np.random.uniform(0, 30)  # Uniform distribution height

        if y < distribution(x):
            accepted_samples.append((x, y))
        else:
            rejected_samples.append((x, y))

    return np.array(accepted_samples), np.array(rejected_samples)
samples = 100
# the point of uniform distribution
a= -2
b= 2
#the data
accepted_samples, rejected_samples = rejection_sampling(samples)

#graph
x_values = np.linspace(-2, 2, 1000)
plt.plot(x_values, distribution(x_values), color= 'red', label='Distribution')

plt.scatter(accepted_samples[:, 0], accepted_samples[:, 1], color='red', label='Accepted Samples', alpha=0.5)
plt.scatter(rejected_samples[:, 0], rejected_samples[:, 1], color='green', label='Rejected Samples', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Rejection Sampling for f(x)')


# <h2>No. 2</h2>

#  <h2>$$2−arctan^2(x)=0$$</h2>
#  <h2>$$arctan(x)^2=2$$</h2>
#  <h2>$$arctan(x)=±\sqrt{2}$$</h2>
#  <h2>$$=±tan(\sqrt{2})$$</h2>

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# the function
def distribution(x):
    return 2 - np.arctan(x)**2
# Uniform distribution
def Uniform_distribution(lower, upper, size):
    return np.random.uniform(lower, upper, size)
# the rejection sampling code
def rejection_sampling(samples):
    accepted_samples = []
    rejected_samples = []

    while len(accepted_samples) < samples:
        x = Uniform_distribution(a, b, 1)[0]
        y = np.random.uniform(0, 2.5)  # Uniform distribution height

        if y < distribution(x):
            accepted_samples.append((x, y))
        else:
            rejected_samples.append((x, y))

    return np.array(accepted_samples), np.array(rejected_samples)
samples = 100
# the points of uniform distribution
a= -np.tan(np.sqrt(2))
b= np.tan(np.sqrt(2))
#the data
accepted_samples, rejected_samples = rejection_sampling(samples)

#graph
x_values = np.linspace(-7, 7, 1000)
plt.plot(x_values, distribution(x_values), color= 'red', label='Distribution')

plt.scatter(accepted_samples[:, 0], accepted_samples[:, 1], color='red', label='Accepted Samples', alpha=0.5)
plt.scatter(rejected_samples[:, 0], rejected_samples[:, 1], color='green', label='Rejected Samples', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Rejection Sampling for f(x)')


# In[ ]:





# In[ ]:





# In[ ]:




