# YOUR CODE HERE
def complete_sequence(N, A):
    while True:
        modified = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                modified = True
                if A[i] < A[i+1]:
                    # Insert increasing sequence
                    new_elements = list(range(A[i]+1, A[i+1]))
                else:
                    # Insert decreasing sequence
                    new_elements = list(range(A[i]-1, A[i+1], -1))
                # Insert new_elements between A[i] and A[i+1]
                A = A[:i+1] + new_elements + A[i+1:]
                break
        if not modified:
            break
    return A

# Read input
N, *rest = map(int, open(0).read().split())
A = rest[:N]

# Complete the sequence
result = complete_sequence(N, A)

# Print the result
print(' '.join(map(str, result)))