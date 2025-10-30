from itertools import permutations
from collections import Counter

def is_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def solve():
    n, k = map(int, input().split())
    s = input()
    
    count = 0
    perms = set(permutations(s))
    
    for p in perms:
        p_str = "".join(p)
        if not is_palindrome(p_str, k):
            count += 1
            
    print(count)

solve()