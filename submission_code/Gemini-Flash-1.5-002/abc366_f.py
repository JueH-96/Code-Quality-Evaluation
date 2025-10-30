# YOUR CODE HERE
import itertools

N, K = map(int, input().split())
AB = []
for i in range(N):
    A, B = map(int, input().split())
    AB.append((A, B))

max_val = 0
for p in itertools.permutations(range(N), K):
    val = 1
    for i in p:
        val = AB[i][0] * val + AB[i][1]
    max_val = max(max_val, val)

print(max_val)