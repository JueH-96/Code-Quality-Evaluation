import sys
from array import array

# ------------------------------------------------------------
#  Shortest walk that visits every vertex at least once
#  (N ≤ 20, M ≤ N(N-1), no negative cycle)
#
#  1.  Floyd-Warshall gives all-pairs shortest distances d.
#  2.  With those distances the problem becomes the classical
#      “shortest Hamiltonian path” on the complete graph whose
#      edge weights are d.  (Any walk can be shrunk to the
#      order of the first visits of every vertex, and that
#      order can be realised by concatenating shortest paths.)
#  3.  DP[mask][v] = minimum cost for a path that has visited
#      vertices in ‘mask’ and ends in ‘v’.
#      size of state space :  N·2^(N-1)  ≤ 10 485 760
#  4.  If every vertex can be reached the answer is
#         min_v DP[(1<<N)-1][v]
#      otherwise “No”.
# ------------------------------------------------------------

def main() -> None:
    input = sys.stdin.buffer.readline
    N, M = map(int, input().split())

    INF = 10 ** 18

    # --- all-pairs shortest distances ------------------------
    d = [[INF] * N for _ in range(N)]
    for i in range(N):
        d[i][i] = 0

    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1
        v -= 1
        if w < d[u][v]:                 # multiple edges safety
            d[u][v] = w

    for k in range(N):
        dk = d[k]
        for i in range(N):
            if d[i][k] == INF:
                continue
            ik = d[i][k]
            di = d[i]
            for j in range(N):
                if dk[j] == INF:
                    continue
                if ik + dk[j] < di[j]:
                    di[j] = ik + dk[j]

    # --- bit DP for shortest Hamiltonian path ----------------
    ALL = 1 << N
    dp = [array('q', [INF]) * ALL for _ in range(N)]
    for v in range(N):
        dp[v][1 << v] = 0

    # helper: index of a power-of-two
    idx_of = {1 << i: i for i in range(N)}
    full_mask = ALL - 1

    for mask in range(ALL):
        sub = mask
        while sub:
            lb = sub & -sub          # one vertex in mask
            v = idx_of[lb]
            cost_here = dp[v][mask]
            if cost_here != INF:
                rem = full_mask ^ mask   # not yet visited
                while rem:
                    lb2 = rem & -rem
                    u = idx_of[lb2]
                    if d[v][u] != INF:
                        newmask = mask | lb2
                        newcost = cost_here + d[v][u]
                        if newcost < dp[u][newmask]:
                            dp[u][newmask] = newcost
                    rem ^= lb2
            sub ^= lb

    answer = min(dp[v][full_mask] for v in range(N))
    if answer == INF:
        print("No")
    else:
        print(answer)


if __name__ == "__main__":
    main()