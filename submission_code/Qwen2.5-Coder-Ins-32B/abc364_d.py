# YOUR CODE HERE
import sys
import bisect

input = sys.stdin.read().split()
N = int(input[0])
Q = int(input[1])
A = list(map(int, input[2:2+N]))
queries = []

index = 2 + N
for _ in range(Q):
    b = int(input[index])
    k = int(input[index + 1])
    queries.append((b, k))
    index += 2

A.sort()

results = []
for b, k in queries:
    distances = [abs(a - b) for a in A]
    distances.sort()
    results.append(distances[k-1])

for result in results:
    print(result)