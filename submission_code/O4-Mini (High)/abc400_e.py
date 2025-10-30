import sys
import math
from bisect import bisect_right

def main():
    data = sys.stdin.buffer.read().split()
    Q = int(data[0])
    A_list = list(map(int, data[1:1+Q]))
    maxA = max(A_list)
    M_max = math.isqrt(maxA)

    # Build smallest-prime-factor array up to M_max
    spf = [0] * (M_max + 1)
    for i in range(2, M_max + 1):
        if spf[i] == 0:
            spf[i] = i
            if i <= M_max // i:
                for j in range(i*i, M_max + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # dp[i] = number of distinct prime factors of i
    dp = [0] * (M_max + 1)
    for i in range(2, M_max + 1):
        p = spf[i]
        x = i // p
        if spf[x] == p:
            dp[i] = dp[x]
        else:
            dp[i] = dp[x] + 1

    # Precompute all 400‑numbers: N = i^2 where i has exactly 2 distinct primes
    N_list = [i*i for i in range(2, M_max + 1) if dp[i] == 2]

    # Answer queries by binary searching the largest N_list[j] <= A
    out = []
    for A in A_list:
        idx = bisect_right(N_list, A) - 1
        out.append(str(N_list[idx]))

    sys.stdout.write("
".join(out))

main()