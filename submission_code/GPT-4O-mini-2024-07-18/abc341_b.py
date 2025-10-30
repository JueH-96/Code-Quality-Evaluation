def max_currency(N, A, operations):
    for i in range(N - 1):
        S, T = operations[i]
        # Calculate how many times we can perform the operation
        while A[i] >= S:
            # Perform the operation
            A[i] -= S
            A[i + 1] += T
    return A[N - 1]

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
A = list(map(int, data[1].split()))
operations = [tuple(map(int, line.split())) for line in data[2:]]

# Get the result and print it
result = max_currency(N, A, operations)
print(result)