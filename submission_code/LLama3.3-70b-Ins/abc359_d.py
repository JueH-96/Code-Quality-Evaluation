import sys
from itertools import product

MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def is_good_string(s, k):
    for i in range(len(s) - k + 1):
        if is_palindrome(s[i:i+k]):
            return False
    return True

def solve(n, k, s):
    count = 0
    for p in product('AB', repeat=s.count('?')):
        t = list(s)
        j = 0
        for i in range(n):
            if t[i] == '?':
                t[i] = p[j]
                j += 1
        t = ''.join(t)
        if is_good_string(t, k):
            count += 1
    return count % MOD

def main():
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    print(solve(n, k, s))

if __name__ == '__main__':
    main()