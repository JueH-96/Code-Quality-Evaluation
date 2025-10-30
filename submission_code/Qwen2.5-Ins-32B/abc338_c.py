import sys

def max_servings(N, Q, A, B):
    max_servings = 0
    for x in range(0, max(Q) + 1):
        for y in range(0, max(Q) + 1):
            if all(x * A[i] + y * B[i] <= Q[i] for i in range(N)):
                max_servings = max(max_servings, x + y)
    return max_servings

input = sys.stdin.read
data = list(map(int, input().split()))
N = data[0]
Q = data[1:N+1]
A = data[N+1:2*N+1]
B = data[2*N+1:]

print(max_servings(N, Q, A, B))