# YOUR CODE HERE
import sys
from itertools import combinations

def max_weight(N, D):
    max_weight = 0
    for subset in combinations(range(N), N//2):
        weight = sum(D[i][j] for i in subset for j in subset if i < j)
        max_weight = max(max_weight, weight)
    return max_weight

N = int(input())
D = [list(map(int, line.split())) for line in sys.stdin.readlines()]
print(max_weight(N, D))