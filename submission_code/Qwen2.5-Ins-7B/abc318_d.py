# YOUR CODE HERE
from itertools import combinations

N = int(input())
D = [list(map(int, input().split())) for _ in range(N)]

max_weight = 0
for i in range(1, N+1):
    for comb in combinations(range(N), i):
        weights = []
        for j in range(i-1):
            for k in range(j+1, i):
                weights.append(D[comb[j]][comb[k]])
        max_weight = max(max_weight, sum(weights))

print(max_weight)