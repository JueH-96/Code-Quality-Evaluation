from itertools import permutations
from collections import Counter

def is_palindrome(s):
    return s == s[::-1]

def count_valid_permutations(N, K, S):
    # Generate all unique permutations
    unique_perms = set(permutations(S))
    count = 0
    for perm in unique_perms:
        # Convert tuple to string
        perm_str = ''.join(perm)
        # Check if any substring of length K is a palindrome
        has_palindrome = False
        for i in range(N - K + 1):
            substring = perm_str[i:i+K]
            if is_palindrome(substring):
                has_palindrome = True
                break
        if not has_palindrome:
            count += 1
    return count

# Read input
N, K = map(int, input().split())
S = input().strip()

# Compute and print the result
print(count_valid_permutations(N, K, S))