import math
from sys import stdin

def main():
    N, K = map(int, stdin.read().split())
    if K == 0:
        print()
        return
    parts = [K] * N
    elements = [i + 1 for i in range(N)]
    current_parts = parts.copy()
    remaining = N * K
    result = []

    # Precompute factorial up to N*K
    max_fact = N * K
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i - 1] * i

    def compute_multinomial(n, parts):
        numerator = fact[n]
        denominator = 1
        for p in parts:
            denominator *= fact[p]
        return numerator // denominator

    # Calculate initial k
    s = compute_multinomial(remaining, current_parts)
    k = (s + 1) // 2

    while remaining > 0:
        for x in range(N):
            if current_parts[x] == 0:
                continue
            new_parts = current_parts.copy()
            new_parts[x] -= 1
            cnt = compute_multinomial(remaining - 1, new_parts)
            if cnt < k:
                k -= cnt
                continue
            else:
                result.append(x + 1)
                current_parts[x] -= 1
                remaining -= 1
                break

    print(' '.join(map(str, result)))

main()