# YOUR CODE HERE
import math

def max_rating(N, P):
    max_R = float('-inf')
    
    for i in range(N):
        sum_weighted_performance = 0
        sum_weights = 0
        for j in range(i, N):
            weight = 0.9 ** (j - i)
            sum_weighted_performance += weight * P[j]
            sum_weights += weight
            k = j - i + 1
            R = sum_weighted_performance / sum_weights - 1200 / math.sqrt(k)
            max_R = max(max_R, R)
    
    return max_R

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:]))

# Calculate and print the result
result = max_rating(N, P)
print(f"{result:.15f}")