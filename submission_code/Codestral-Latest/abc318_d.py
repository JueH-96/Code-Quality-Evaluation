import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
D = [[0] * (N+1) for _ in range(N+1)]

index = 1
for i in range(1, N):
    for j in range(i+1, N+1):
        D[i][j] = int(data[index])
        D[j][i] = D[i][j]
        index += 1

max_weight = 0

for k in range(1, N+1):
    for comb in combinations(range(1, N+1), k):
        if all(i not in comb or j not in comb for i, j in combinations(comb, 2)):
            total_weight = sum(D[i][j] for i, j in combinations(comb, 2))
            max_weight = max(max_weight, total_weight)

print(max_weight)