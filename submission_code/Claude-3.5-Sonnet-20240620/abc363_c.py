# YOUR CODE HERE
from itertools import permutations
from collections import Counter

def contains_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_valid_permutations(s, k):
    counter = Counter(s)
    total_perms = sum(1 for _ in permutations(s))
    invalid_perms = 0

    for perm in permutations(s):
        if contains_palindrome(perm, k):
            invalid_perms += 1

    return total_perms - invalid_perms

n, k = map(int, input().split())
s = input().strip()

result = count_valid_permutations(s, k)
print(result)