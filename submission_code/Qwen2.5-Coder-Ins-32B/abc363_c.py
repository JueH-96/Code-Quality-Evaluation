# YOUR CODE HERE
from itertools import permutations

def contains_palindrome_of_length_k(s, k):
    n = len(s)
    for i in range(n - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_valid_permutations(s, k):
    unique_permutations = set(permutations(s))
    valid_count = 0
    for perm in unique_permutations:
        perm_str = ''.join(perm)
        if not contains_palindrome_of_length_k(perm_str, k):
            valid_count += 1
    return valid_count

import sys
input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
S = input[2]

result = count_valid_permutations(S, K)
print(result)