# YOUR CODE HERE
from math import pi

N = int(input())
pi_str = str(pi)
result = pi_str[:4] + pi_str[7:N+7]
print(result)