import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

index = 0
T = int(data[index])
index += 1
results = []

for _ in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    index += 2
    A = list(map(int, data[index:index + N]))
    index += N
    B = list(map(int, data[index:index + N]))
    index += N

    min_value = float('inf')

    for comb in combinations(range(N), K):
        max_A = max(A[i] for i in comb)
        sum_B = sum(B[i] for i in comb)
        min_value = min(min_value, max_A * sum_B)

    results.append(min_value)

for result in results:
    print(result)