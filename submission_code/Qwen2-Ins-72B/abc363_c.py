from itertools import permutations
from collections import Counter

def is_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def solve(n, k, s):
    unique_perms = set(permutations(s))
    valid_perms = [p for p in unique_perms if not is_palindrome(''.join(p), k)]
    return len(valid_perms)

if __name__ == "__main__":
    n, k = map(int, input().split())
    s = input()
    print(solve(n, k, s))