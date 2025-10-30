import sys

def max_score(A):
    N = len(A)
    score = 0
    while N > 1:
        max_diff = 0
        max_idx = -1
        for i in range(N-1):
            diff = abs(A[i] - A[i+1])
            if diff > max_diff:
                max_diff = diff
                max_idx = i
        score += max_diff
        A = A[:max_idx] + A[max_idx+2:]
        N -= 1
    return score

# Read input from stdin
N = int(input())
A = [int(x) for x in input().split()]

# Compute the maximum score
max_score_value = max_score(A)

# Write output to stdout
print(max_score_value)