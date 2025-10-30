# YOUR CODE HERE
n, k = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Find the cycle length
cycle = [0] * (n + 1)
for i in range(n):
    if cycle[X[i]] == 0:
        j = i
        while cycle[X[j]] == 0:
            cycle[X[j]] = j + 1
            j = X[j] - 1
        cycle_length = j - X[j] + 1
        break

# Determine the effective number of operations
if k >= cycle_length:
    k %= cycle_length

# Perform the operations
for _ in range(k):
    for i in range(n):
        A[i], A[X[i] - 1] = A[X[i] - 1], A[i]

# Print the result
print(*A)