# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:2*N+1]))
B = list(map(int, data[2*N+1:]))

max_servings = 0

for a in range(Q[0] // A[0] + 1):
    for b in range(Q[0] // B[0] + 1):
        if all(Q[i] >= a * A[i] + b * B[i] for i in range(N)):
            max_servings = max(max_servings, a + b)

print(max_servings)