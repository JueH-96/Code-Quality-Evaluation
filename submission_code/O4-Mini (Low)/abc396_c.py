import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    B = [int(next(it)) for _ in range(n)]
    W = [int(next(it)) for _ in range(m)]

    # Sort black balls descending and compute prefix sums
    B.sort(reverse=True)
    prefixB = [0] * (n + 1)
    for i in range(n):
        prefixB[i+1] = prefixB[i] + B[i]

    # Count how many black balls have positive value
    # Since sorted descending, all B[0..P-1] > 0, B[P] <= 0
    P = 0
    while P < n and B[P] > 0:
        P += 1
    sum_positive_blacks = prefixB[P]  # sum of all positive black balls

    # Sort white balls descending and compute prefix sums
    W.sort(reverse=True)
    # We only need up to min(m, n) whites, since can't pick more whites than black
    K = min(m, n)
    prefixW = [0] * (K + 1)
    for i in range(K):
        prefixW[i+1] = prefixW[i] + W[i]

    # Try picking w whites for w = 0..K
    # For each w, we must pick at least w blacks.
    # Optimal black pick is:
    #   if w <= P: take all positive blacks (sum_positive_blacks)
    #   else: take top w blacks (which may include some non-positive)
    ans = 0
    for w in range(K + 1):
        if w <= P:
            black_sum = sum_positive_blacks
        else:
            black_sum = prefixB[w]
        total = prefixW[w] + black_sum
        if total > ans:
            ans = total

    print(ans)

if __name__ == "__main__":
    main()