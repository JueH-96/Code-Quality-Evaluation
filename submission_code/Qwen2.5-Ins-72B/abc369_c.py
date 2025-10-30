# YOUR CODE HERE
import sys

def is_arithmetic_progression(subseq):
    if len(subseq) <= 1:
        return True
    diff = subseq[1] - subseq[0]
    for i in range(1, len(subseq)):
        if subseq[i] - subseq[i-1] != diff:
            return False
    return True

def count_arithmetic_subsequences(N, A):
    count = 0
    for l in range(N):
        for r in range(l, N):
            if is_arithmetic_progression(A[l:r+1]):
                count += 1
    return count

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve the problem
result = count_arithmetic_subsequences(N, A)

# Print the result
print(result)