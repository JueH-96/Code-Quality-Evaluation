# Read input
S = input().strip()
N = int(input())

L = len(S)
max_T = int(S.replace('?', '1'), 2)

if max_T <= N:
    print(max_T)
    exit()

B = bin(N)[2:].zfill(L)

from functools import lru_cache

@lru_cache(maxsize=None)
def dp(i, tight):
    if i == L:
        return 0
    current_bit = S[i]
    if current_bit in ('0', '1'):
        bit = int(current_bit)
        if tight:
            if bit > int(B[i]):
                return -1
            new_tight = tight and (bit == int(B[i]))
            res = dp(i + 1, new_tight)
            if res == -1:
                return -1
            return (bit << (L - i - 1)) + res
        else:
            res = dp(i + 1, False)
            if res == -1:
                return -1
            return (bit << (L - i - 1)) + res
    else:
        max_val = -1
        if tight:
            max_possible = int(B[i])
            if 1 <= max_possible:
                new_tight = tight and (1 == max_possible)
                res = dp(i + 1, new_tight)
                if res != -1:
                    val = (1 << (L - i - 1)) + res
                    max_val = val
            new_tight = tight and (0 == max_possible)
            res = dp(i + 1, new_tight)
            if res != -1:
                val = (0 << (L - i - 1)) + res
                if val > max_val:
                    max_val = val
        else:
            res = dp(i + 1, False)
            if res != -1:
                max_val = (1 << (L - i - 1)) + res
        return max_val

result = dp(0, True)
print(result if result != -1 else -1)