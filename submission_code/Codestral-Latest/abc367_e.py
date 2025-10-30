# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
K = int(data[index + 1])
index += 2

X = list(map(int, data[index:index + N]))
index += N
A = list(map(int, data[index:index + N]))

# If K is 0, no operations are performed
if K == 0:
    print(" ".join(map(str, A)))
else:
    # Find the cycle length
    visited = [False] * N
    cycle_length = 0
    current = 0
    while not visited[current]:
        visited[current] = True
        cycle_length += 1
        current = X[current] - 1

    # Reduce K to the effective number of operations within the cycle
    K %= cycle_length

    # Perform the remaining operations
    for _ in range(K):
        A = [A[X[i] - 1] for i in range(N)]

    print(" ".join(map(str, A)))