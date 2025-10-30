from collections import defaultdict

def max_1122_subarray(A):
    n = len(A)
    if n % 2 != 0:
        return 0

    freq = defaultdict(list)
    for i, a in enumerate(A):
        freq[a].append(i)

    max_len = 0
    for a, indices in freq.items():
        if len(indices) % 2 != 0:
            continue
        for i in range(0, len(indices), 2):
            start = indices[i]
            end = indices[i+1] + 1
            length = end - start
            max_len = max(max_len, length)

    return max_len

# Read input
N = int(input())
A = [int(x) for x in input().split()]

# Solve the problem
result = max_1122_subarray(A)

# Write output
print(result)