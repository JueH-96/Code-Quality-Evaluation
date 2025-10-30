import sys

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    INF = M + 1          # value larger than any R_i (1 ≤ R_i ≤ M)

    # bestR[x] = smallest R among intervals whose left end is exactly x
    bestR = [INF] * (M + 2)
    for _ in range(N):
        L = int(next(it))
        R = int(next(it))
        if R < bestR[L]:
            bestR[L] = R

    # minR[l] = minimum R among intervals with L ≥ l
    minR = [INF] * (M + 3)
    minR[M + 1] = INF
    for l in range(M, 0, -1):
        if bestR[l] < minR[l + 1]:
            minR[l] = bestR[l]
        else:
            minR[l] = minR[l + 1]

    # count valid pairs
    ans = 0
    for l in range(1, M + 1):
        limit = minR[l] - 1         # largest allowed r for this l
        # (minR[l]==INF) ⇒ limit = M
        if limit > M:
            limit = M
        if limit >= l:
            ans += limit - l + 1

    print(ans)

if __name__ == "__main__":
    main()