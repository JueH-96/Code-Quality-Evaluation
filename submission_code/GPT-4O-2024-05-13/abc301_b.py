# YOUR CODE HERE
def process_sequence(N, A):
    i = 0
    while i < len(A) - 1:
        if abs(A[i] - A[i + 1]) != 1:
            if A[i] < A[i + 1]:
                to_insert = list(range(A[i] + 1, A[i + 1]))
            else:
                to_insert = list(range(A[i] - 1, A[i + 1], -1))
            A = A[:i + 1] + to_insert + A[i + 1:]
            i = 0  # Restart from the beginning
        else:
            i += 1
    return A

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Process the sequence
result = process_sequence(N, A)

# Print the result
print(" ".join(map(str, result)))