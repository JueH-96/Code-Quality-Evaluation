import sys
from input import input

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

A = list(map(int, data[index:index + N]))
index += N

queries = []
for _ in range(Q):
    b_j = int(data[index])
    k_j = int(data[index + 1])
    queries.append((b_j, k_j))
    index += 2

for b_j, k_j in queries:
    distances = sorted(abs(a - b_j) for a in A)
    print(distances[k_j - 1])