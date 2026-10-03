import numpy as np

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

mean = np.mean(speed)
std = np.std(speed)  # standard deviation
var = np.var(speed)  # variance: std^2

# a number that describes the value that a given percent of the values are lower than.
percentile = np.percentile(speed, 100)

print(mean)
print(std)
print(var)
print(pow(std, 2))
print(percentile)
