import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    MOD = 998244353
    # maxL[pos][r]: maximum L among constraints (L,R,pos) with R <= r
    # 1-indexed
    maxL = [ [0] * (N+1) for _ in range(N+1) ]
    for _ in range(M):
        L = int(next(it))
        R = int(next(it))
        X = int(next(it))
        # constraint forbids X as root in any segment [l,r] with l<=L and r>=R
        # we record for pos=X at r=R the L
        if L > maxL[X][R]:
            maxL[X][R] = L
    # prefix max over R for each pos
    for pos in range(1, N+1):
        row = maxL[pos]
        mx = 0
        for r in range(1, N+1):
            if row[r] > mx:
                mx = row[r]
            else:
                row[r] = mx
            # now row[r] == max_{R'<=r} L among constraints for pos
    # dp[l][r]: number of valid permutations in segment [l..r]
    # We'll use 1-indexed, dp size (N+2)x(N+2)
    dp = [ [0] * (N+2) for _ in range(N+2) ]
    # base: empty segments
    for i in range(1, N+2):
        dp[i][i-1] = 1
    # compute by increasing length
    for length in range(1, N+1):
        for l in range(1, N - length + 2):
            r = l + length - 1
            total = 0
            # try each pos as root
            # pos is valid if no constraint (L_i,R_i,pos) fully inside [l,r],
            # i.e. maxL[pos][r] < l
            for pos in range(l, r+1):
                if maxL[pos][r] < l:
                    total += dp[l][pos-1] * dp[pos+1][r]
            dp[l][r] = total % MOD
    # answer for [1,N]
    print(dp[1][N] % MOD)

if __name__ == "__main__":
    main()