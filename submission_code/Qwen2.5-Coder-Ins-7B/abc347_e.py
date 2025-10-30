# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
queries = data[2:]

A = [0] * N
S = set()
size_S = 0

for query in queries:
    x = int(query) - 1
    if x in S:
        S.remove(x)
        size_S -= 1
    else:
        S.add(x)
        size_S += 1
    for j in S:
        A[j] += size_S

for a in A:
    print(a, end=' ')