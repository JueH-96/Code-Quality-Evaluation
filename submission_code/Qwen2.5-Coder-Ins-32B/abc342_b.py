# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
P = list(map(int, input[1:N+1]))
Q = int(input[N+1])
queries = [(int(input[N+2+2*i]), int(input[N+3+2*i])) for i in range(Q)]

positions = {P[i]: i for i in range(N)}

results = []
for A, B in queries:
    if positions[A] < positions[B]:
        results.append(A)
    else:
        results.append(B)

for result in results:
    print(result)