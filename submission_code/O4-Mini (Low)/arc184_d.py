import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    N = int(sys.stdin.readline())
    pts = [None]*N
    for i in range(N):
        x,y = map(int,sys.stdin.readline().split())
        pts[i]=(x,y,i)
    # Re‐index so that X‐coordinates are 1..N in order
    pts.sort()
    # Build P: P[i]=Y‐rank of the point with X‐rank i
    Ys = [y for x,y,_ in pts]
    # compress Ys to 1..N
    Ysorted = sorted(Ys)
    rank = {v:i+1 for i,v in enumerate(Ysorted)}
    P = [0]*(N+1)
    for i,(x,y,orig) in enumerate(pts, start=1):
        P[i]=rank[y]

    # We build DP over intervals [l..r] in the X‐ordering.
    # Let dp[l][r] = number of ways to choose a dominating set
    # for the induced subgraph on {l..r}, assuming there is
    # NO outside vertex to cover any vertex in [l..r].
    # In particular every vertex in [l..r] must be either in S
    # or have a neighbor in [l..r].
    #
    # We use the well‐known “minimal Y” split for permutation graphs.
    # The minimal‐Y in [l..r] is at position m = argmin P[l..r].
    # That vertex m only has comparabilities (neighbors) among
    # those i>m with P[i]>P[m].  It has NO neighbors among i<m.
    # Hence in any dominating‐set of [l..r]:
    #   – If we do NOT pick m, then m must be covered by someone
    #     to its right in S.  And the left side [l..m−1] must
    #     cover themselves internally.
    #   – If we DO pick m, then it covers itself and covers
    #     all i>m with P[i]>P[m], but covers nobody on the left.
    # We split at m into two independent subproblems: left=[l..m−1],
    # right=[m+1..r].  But there is a coupling:
    #   * When m∉S, we REQUIRE that the right‐subinterval
    #     contributes at least one member j>m with P[j]>P[m]
    #     to cover m.
    #   * When m∈S, we impose no further requirement on the right.
    #
    # So we maintain for each interval [l..r]:
    #   dp[l][r][0] = # ways to dominate [l..r] with NO external help,
    #                 AND requiring that the right side MUST
    #                 contribute at least one node whose P> P[m]
    #                 at the split point.
    #   dp[l][r][1] = # ways to dominate [l..r] with NO external help,
    #                 with NO extra requirement on the right.
    #
    # The answer on [1..N] with no external help and no
    # extra requirement is dp[1][N][1].

    dp0 = [[0]*(N+2) for _ in range(N+2)]
    dp1 = [[0]*(N+2) for _ in range(N+2)]
    # empty interval [i..i−1] is “already done” with one way
    # it has nothing to cover
    for i in range(1,N+2):
        dp0[i][i-1] = 1
        dp1[i][i-1] = 1

    # Precompute min‐Y positions for every [l..r]
    minpos = [[0]*(N+1) for _ in range(N+1)]
    for l in range(1,N+1):
        mn = l
        for r in range(l, N+1):
            if P[r] < P[mn]:
                mn = r
            minpos[l][r] = mn

    for length in range(1, N+1):
        for l in range(1, N-length+2):
            r = l+length-1
            m = minpos[l][r]
            # split: left=[l..m−1], right=[m+1..r]
            # 1) m NOT in S:
            #    left must cover itself: use dp1[l][m-1]
            #    right must both cover itself and supply at least
            #      one node j with P[j]>P[m] to cover m:
            #    so number of ways = dp1[l][m-1] * dp0[m+1][r]
            ways_no = dp1[l][m-1] * dp0[m+1][r] % MOD
            # 2) m IN S:
            #    left must cover itself: dp1[l][m-1]
            #    right must cover itself (but no extra requirement): dp1[m+1][r]
            ways_yes = dp1[l][m-1] * dp1[m+1][r] % MOD

            # dp1 = total ways (no extra requirement on right)
            dp1[l][r] = (ways_no + ways_yes) % MOD
            # dp0 = ways in which right must supply at least
            # one cover‐node for the *parent* m
            # Only the “m NOT in S” case imposes that:
            dp0[l][r] = ways_no

    print(dp1[1][N] % MOD)

if __name__ == "__main__":
    main()