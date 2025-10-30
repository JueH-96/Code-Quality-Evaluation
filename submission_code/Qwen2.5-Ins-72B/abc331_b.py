# YOUR CODE HERE
import sys

def min_cost(N, S, M, L):
    min_cost = float('inf')
    
    for i in range(N // 6 + 1):
        for j in range(N // 8 + 1):
            for k in range(N // 12 + 1):
                if 6 * i + 8 * j + 12 * k >= N:
                    cost = S * i + M * j + L * k
                    min_cost = min(min_cost, cost)
    
    return min_cost

N, S, M, L = map(int, sys.stdin.readline().strip().split())
print(min_cost(N, S, M, L))