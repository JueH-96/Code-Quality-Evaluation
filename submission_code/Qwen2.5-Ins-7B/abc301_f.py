# YOUR CODE HERE
from math import pow

MOD = 998244353

def count_valid_strings(s):
    q = s.count('?')
    if q == 0:
        return 1 if not is_ddos_type(s) else 0
    
    total = pow(52, q, MOD)
    invalid = 0
    
    for i in range(len(s)):
        if s[i] == '?':
            for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                new_s = s[:i] + c + s[i+1:]
                if not is_ddos_type(new_s):
                    invalid += pow(52, q-1, MOD)
                    invalid %= MOD
    
    return (total - invalid + MOD) % MOD

def is_ddos_type(s):
    if len(s) != 4:
        return False
    if s[0].islower() or s[1].islower() or s[3].islower():
        return False
    if s[0] != s[1]:
        return False
    return True

s = input().strip()
print(count_valid_strings(s))