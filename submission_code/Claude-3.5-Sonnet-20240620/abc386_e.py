# YOUR CODE HERE
from itertools import combinations

def max_xor(arr, k):
    max_val = 0
    for combo in combinations(arr, k):
        xor = 0
        for num in combo:
            xor ^= num
        max_val = max(max_val, xor)
    return max_val

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Calculate and print the result
result = max_xor(A, K)
print(result)