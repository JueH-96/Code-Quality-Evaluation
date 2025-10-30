from functools import lru_cache

MOD = 998244353

@lru_cache(maxsize=None)
def count_strings(k, counts):
    if k == 0:
        return 1
    
    total = 0
    for i in range(26):
        if counts[i] > 0:
            counts[i] -= 1
            total += count_strings(k - 1, counts)
            total %= MOD
            counts[i] += 1
    
    return total

def solve():
    k = int(input())
    counts = [int(x) for x in input().split()]
    return count_strings(k, tuple(counts))

print(solve())