# YOUR CODE HERE
import sys

def calculate_cost(N):
    cost = 0
    while N > 1:
        cost += N
        N -= 1
    return cost

N = int(input())
print(calculate_cost(N))