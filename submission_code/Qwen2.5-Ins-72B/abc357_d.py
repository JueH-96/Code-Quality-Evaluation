# YOUR CODE HERE
import sys

def mod_exp(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def find_remainder(N, mod=998244353):
    N_str = str(N)
    len_N = len(N_str)
    total_length = N * len_N
    base = int(N_str)
    result = 0
    current_power = 1
    for _ in range(len_N):
        result = (result + base * current_power * (N % mod)) % mod
        current_power = (current_power * 10 ** len_N) % mod
    return result

N = int(input().strip())
print(find_remainder(N))