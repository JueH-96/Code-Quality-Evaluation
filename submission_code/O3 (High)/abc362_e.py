import sys
from collections import defaultdict

MOD = 998244353


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # answer[k] : number of arithmetic subsequences of length k  (1-indexed)
    ans = [0] * (N + 1)
    ans[1] = N % MOD          # every single element

    # dp[i] :  dict  diff -> dict(length -> count)
    #          all arithmetic subsequences that end at position i (0-based),
    #          have common difference `diff` and given length (length ≥ 2)
    dp = [dict() for _ in range(N)]

    for i in range(N):
        for j in range(i):
            diff = A[i] - A[j]

            # dictionary for this (i, diff)
            rec_i = dp[i].get(diff)
            if rec_i is None:
                rec_i = {}
                dp[i][diff] = rec_i

            # 1) the new subsequence consisting of the pair (j, i)
            rec_i[2] = (rec_i.get(2, 0) + 1) % MOD
            ans[2] = (ans[2] + 1) % MOD

            # 2) extend every subsequence that ends at j with the same diff
            rec_j = dp[j].get(diff)
            if rec_j:
                for length, cnt in rec_j.items():      # length ≥ 2
                    new_len = length + 1
                    rec_i[new_len] = (rec_i.get(new_len, 0) + cnt) % MOD
                    ans[new_len] = (ans[new_len] + cnt) % MOD

    # output
    print(' '.join(str(ans[k] % MOD) for k in range(1, N + 1)))


if __name__ == "__main__":
    main()