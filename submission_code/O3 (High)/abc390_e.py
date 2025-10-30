import sys

def knapsack(items, X):
    """
    0/1 knapsack:
    dp[w] = maximum vitamin units that can be obtained
           with total calories at most w (using only `items`)
    """
    dp = [0] * (X + 1)
    for a, c in items:          # a : vitamin units, c : calories
        if c > X:
            continue
        for w in range(X, c - 1, -1):
            nv = dp[w - c] + a
            if nv > dp[w]:
                dp[w] = nv
    # make it monotone: with more calories you can never do worse
    for w in range(1, X + 1):
        if dp[w - 1] > dp[w]:
            dp[w] = dp[w - 1]
    return dp


def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    X = int(next(it))

    vit_items = {1: [], 2: [], 3: []}          # lists of (A_i, C_i)
    for _ in range(N):
        v = int(next(it))
        a = int(next(it))
        c = int(next(it))
        vit_items[v].append((a, c))

    # knapsack for each vitamin
    dp1 = knapsack(vit_items[1], X)
    dp2 = knapsack(vit_items[2], X)
    dp3 = knapsack(vit_items[3], X)

    ans = 0
    # enumerate calories used by vitamin-1 part
    for c1 in range(X + 1):
        s1 = dp1[c1]
        if s1 <= ans:        # even with perfect other vitamins we cannot beat ans
            continue
        rem_after_1 = X - c1
        # enumerate calories for vitamin-2 part
        for c2 in range(rem_after_1 + 1):
            s2 = dp2[c2]
            if s2 <= ans:
                continue
            s3 = dp3[rem_after_1 - c2]
            # best achievable minimum with this split
            m = min(s1, s2, s3)
            if m > ans:
                ans = m
                # early stopping is possible only if ans == s1
                # but savings are tiny, so we skip further tricks
    print(ans)


if __name__ == "__main__":
    main()