# YOUR CODE HERE
MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    q = S.count('?')
    total_combinations = 2 ** q
    good_count = 0

    from itertools import product

    for replacement in product('AB', repeat=q):
        t = list(S)
        idx = 0
        for i in range(N):
            if t[i] == '?':
                t[i] = replacement[idx]
                idx += 1
        t = ''.join(t)
        
        is_good = True
        for i in range(N - K + 1):
            if is_palindrome(t[i:i+K]):
                is_good = False
                break
        
        if is_good:
            good_count += 1

    return good_count % MOD

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
S = data[2]

print(count_good_strings(N, K, S))