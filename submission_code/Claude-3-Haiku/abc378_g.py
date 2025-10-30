from itertools import permutations
from bisect import bisect_left

def longest_increasing_subsequence(arr):
    """
    Find the length of the longest increasing subsequence in the given array.
    """
    n = len(arr)
    dp = [float('inf')] * (n + 1)
    dp[0] = -float('inf')
    for x in arr:
        i = bisect_left(dp, x)
        dp[i] = x
    return sum(dp[i] != float('inf') for i in range(n + 1))

def longest_decreasing_subsequence(arr):
    """
    Find the length of the longest decreasing subsequence in the given array.
    """
    return longest_increasing_subsequence([-x for x in arr])

def count_permutations(A, B, M):
    """
    Count the number of permutations satisfying the given conditions, modulo M.
    """
    count = 0
    for p in permutations(range(1, A * B)):
        if (longest_increasing_subsequence(p) == A and
            longest_decreasing_subsequence(p) == B and
            any(longest_increasing_subsequence(p + (n + 0.5,)) == A and
                longest_decreasing_subsequence(p + (n + 0.5,)) == B for n in range(A * B))):
            count += 1
    return count % M

# Read input from stdin
A, B, M = map(int, input().split())

# Solve the problem and write the answer to stdout
print(count_permutations(A, B, M))