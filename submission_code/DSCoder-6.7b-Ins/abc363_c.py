import sys
from itertools import permutations

def has_palindrome(s, k):
    for i in range(len(s)-k+1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def solve(n, k, s):
    perms = set(permutations(s, n))
    count = 0
    for p in perms:
        if not has_palindrome(''.join(p), k):
            count += 1
    return count

n, k = map(int, sys.stdin.readline().split())
s = sys.stdin.readline().strip()
print(solve(n, k, s))