from typing import List

def count_arithmetic_subsequences(A: List[int]) -> int:
    n = len(A)
    count = 0

    for l in range(1, n+1):
        for r in range(l, n+1):
            if is_arithmetic_progression(A[l-1:r]):
                count += 1

    return count

def is_arithmetic_progression(seq: List[int]) -> bool:
    if len(seq) <= 1:
        return True

    d = seq[1] - seq[0]
    for i in range(1, len(seq)):
        if seq[i] - seq[i-1] != d:
            return False

    return True

# Read input
n = int(input())
A = [int(x) for x in input().split()]

# Solve the problem
result = count_arithmetic_subsequences(A)

# Print the answer
print(result)