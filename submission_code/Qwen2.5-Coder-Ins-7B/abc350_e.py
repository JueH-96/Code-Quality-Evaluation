import sys
import math

def min_cost(N, A, X, Y):
    if N == 0:
        return 0
    if A == 1:
        return X * N
    if Y == 0:
        return X * N
    if X == 0:
        return Y * math.log(N, A)
    
    # Calculate the expected cost of rolling the die
    die_cost = Y * (1/6 * (min_cost(N//1, A, X, Y) + min_cost(N//2, A, X, Y) + min_cost(N//3, A, X, Y) + min_cost(N//4, A, X, Y) + min_cost(N//5, A, X, Y) + min_cost(N//6, A, X, Y)))
    
    # Calculate the cost of replacing N with floor(N/A)
    replace_cost = X + min_cost(N//A, A, X, Y)
    
    # Return the minimum of the two costs
    return min(die_cost, replace_cost)

# Read input
N, A, X, Y = map(int, sys.stdin.readline().split())

# Calculate and print the minimum cost
print(min_cost(N, A, X, Y))