import itertools
from collections import Counter

def is_palindrome(s):
    return s == s[::-1]

def contains_palindrome(s, k):
    for i in range(len(s) - k + 1):
        sub = s[i:i+k]
        if is_palindrome(sub):
            return True
    return False

def solve():
    n, k = map(int, input().split())
    s = input()
    
    count = 0
    perms = set(itertools.permutations(s))
    
    for perm in perms:
        perm_str = "".join(perm)
        if not contains_palindrome(perm_str, k):
            count += 1
            
    print(count)

solve()