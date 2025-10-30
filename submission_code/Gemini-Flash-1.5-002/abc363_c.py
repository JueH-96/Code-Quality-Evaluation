# YOUR CODE HERE
from itertools import permutations

def is_palindrome(s):
    return s == s[::-1]

def contains_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if is_palindrome(s[i:i+k]):
            return True
    return False

n, k = map(int, input().split())
s = input()

count = 0
unique_permutations = set()

for p in permutations(s):
    perm_str = "".join(p)
    if not contains_palindrome(perm_str, k):
        count += 1

print(count)