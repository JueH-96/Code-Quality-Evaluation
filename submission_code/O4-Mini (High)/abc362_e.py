import sys
def main():
    mod = 998244353
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # dp[i] will be a dict: diff -> list f where f[l] is the count of arithmetic
    # subsequences of length l ending at index i with common difference 'diff'.
    dp = [dict() for _ in range(N)]
    # ans[k] will accumulate the total number of arithmetic subsequences of length k
    ans = [0] * (N + 1)
    ans[1] = N  # every single element is a subsequence of length 1

    for i in range(N):
        Ai = A[i]
        dp_i = dp[i]
        # consider all ways to end an arithmetic subsequence at i by picking
        # a previous j < i as the penultimate element
        for j in range(i):
            d = Ai - A[j]
            # f will accumulate counts for subsequences ending at i with diff d
            f = dp_i.get(d)
            if f is None:
                # we only need lengths up to N
                f = [0] * (N + 1)
                dp_i[d] = f

            # every pair (j, i) is an arithmetic subsequence of length 2
            f2 = f[2] + 1
            if f2 >= mod:
                f2 -= mod
            f[2] = f2
            ans[2] += 1
            if ans[2] >= mod:
                ans[2] -= mod

            # extend any arithmetic subsequence ending at j with the same diff
            g = dp[j].get(d)
            if g is not None:
                # we only need to check lengths up to N-1 (so that l+1 <= N)
                for l in range(2, N):
                    val = g[l]
                    if val:
                        nl = l + 1
                        fnl = f[nl] + val
                        if fnl >= mod:
                            fnl -= mod
                        f[nl] = fnl
                        ans[nl] += val
                        if ans[nl] >= mod:
                            ans[nl] -= mod

    # output answers for k=1..N
    out = " ".join(str(ans[k] % mod) for k in range(1, N + 1))
    sys.stdout.write(out)

main()