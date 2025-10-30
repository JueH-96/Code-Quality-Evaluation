import sys
import math


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    P = list(map(int, data[1:]))

    NEG_INF = -1e100                 # sufficiently small number
    dp = [NEG_INF] * (N + 2)         # dp[l] = best numerator with length l
    dp[0] = 0.0
    current_len = 0

    for x in P:
        # build subsequences ending with the current contest
        for l in range(current_len, -1, -1):    # iterate backwards
            if dp[l] < -1e90:                   # unreachable length
                continue
            cand = 0.9 * dp[l] + x              # 0.9 * old_numerator + x
            if cand > dp[l + 1]:
                dp[l + 1] = cand
        current_len += 1

    best = NEG_INF
    for k in range(1, N + 1):
        if dp[k] < -1e90:                       # length k not attainable
            continue
        s_k = (1.0 - math.pow(0.9, k)) / 0.1    # sum_{i=0}^{k-1} 0.9^i
        average = dp[k] / s_k
        rating = average - 1200.0 / math.sqrt(k)
        if rating > best:
            best = rating

    # print with enough precision
    print("{:.15f}".format(best))


if __name__ == "__main__":
    main()