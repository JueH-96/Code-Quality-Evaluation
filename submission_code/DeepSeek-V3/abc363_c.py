from itertools import permutations
from collections import Counter

def is_palindrome_substring(s, k):
    n = len(s)
    for i in range(n - k + 1):
        substring = s[i:i+k]
        if substring == substring[::-1]:
            return True
    return False

def count_valid_permutations(N, K, S):
    unique_perms = set([''.join(p) for p in permutations(S)])
    count = 0
    for perm in unique_perms:
        if not is_palindrome_substring(perm, K):
            count += 1
    return count

# Read input
N, K = map(int, input().split())
S = input().strip()

# Compute and print the result
print(count_valid_permutations(N, K, S))