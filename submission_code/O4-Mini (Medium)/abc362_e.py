def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = 998244353

    # dp[i] maps difference d to an array arr where
    # arr[l] = number of arithmetic subsequences of length l ending at i with difference d
    dp = [dict() for _ in range(N)]
    # ans[k] = total number of arithmetic subsequences of length k
    ans = [0] * (N + 1)
    ans[1] = N % M

    for i in range(N):
        ai = A[i]
        for j in range(i):
            d = ai - A[j]
            # get or create dp[i][d]
            dpi = dp[i]
            arr_i = dpi.get(d)
            if arr_i is None:
                arr_i = [0] * (N + 1)
                dpi[d] = arr_i

            # every pair (j,i) forms a length-2 arithmetic subsequence
            arr_i[2] += 1
            if arr_i[2] >= M:
                arr_i[2] -= M
            ans[2] = (ans[2] + 1) % M

            # extend any arithmetic subsequences ending at j with same d
            arr_j = dp[j].get(d)
            if arr_j is not None:
                for l in range(2, N):
                    cnt = arr_j[l]
                    if cnt:
                        nl = l + 1
                        arr_i[nl] = (arr_i[nl] + cnt) % M
                        ans[nl] = (ans[nl] + cnt) % M

    # output answers for k = 1..N
    out = ' '.join(str(ans[k] % M) for k in range(1, N + 1))
    sys.stdout.write(out)


if __name__ == "__main__":
    main()