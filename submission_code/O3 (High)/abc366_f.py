import sys
from math import inf

# -------------------------------------------------------------

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))

    funcs = []
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        # key for sorting (larger key --> inner)
        key = inf if a == 1 else b / (a - 1)
        funcs.append((key, a, b))

    # sort by descending key (bigger key first → inner first)
    funcs.sort(key=lambda x: -x[0])

    # dp[j] = best value after choosing j inner functions
    dp = [1] + [-10**100] * K      # negative large sentinel

    for _, a, b in funcs:
        # update backwards to avoid using the same function twice
        for j in range(min(K - 1, len(dp) - 2), -1, -1):
            if dp[j] > -10**90:       # state reachable?
                cand = a * dp[j] + b
                if cand > dp[j + 1]:
                    dp[j + 1] = cand

    # answer is best value after exactly K functions
    print(int(dp[K]))

# -------------------------------------------------------------
if __name__ == "__main__":
    main()