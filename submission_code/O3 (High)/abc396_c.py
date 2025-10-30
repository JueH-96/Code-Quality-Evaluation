import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    blacks = [int(next(it)) for _ in range(N)]
    whites = [int(next(it)) for _ in range(M)]

    # Sort values in non-increasing order
    blacks.sort(reverse=True)
    whites.sort(reverse=True)

    # Prefix sums of the sorted lists
    pref_b = [0]*(N + 1)
    for i, v in enumerate(blacks, 1):
        pref_b[i] = pref_b[i-1] + v

    pref_w = [0]*(M + 1)
    for i, v in enumerate(whites, 1):
        pref_w[i] = pref_w[i-1] + v

    # best_w[t] = maximum prefix sum using at most t white balls
    best_w = [0]*(M + 1)
    best = 0
    for t in range(1, M + 1):
        best = max(best, pref_w[t])
        best_w[t] = best

    ans = 0  # choosing nothing is always allowed
    for b in range(1, N + 1):
        w_limit = min(b, M)
        total = pref_b[b] + best_w[w_limit]
        if total > ans:
            ans = total

    print(ans)

if __name__ == "__main__":
    main()