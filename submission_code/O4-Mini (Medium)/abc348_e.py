import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a,b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    C = [0] + list(map(int, input().split()))

    # total weight
    S = sum(C)

    # subtree weights
    W = [0]*(N+1)
    # f at root 1
    f1 = 0

    # first dfs: compute W[u] and f1 = sum C[i]*depth(i)
    def dfs1(u, p, depth):
        nonlocal f1
        W[u] = C[u]
        f1 += C[u] * depth
        for v in adj[u]:
            if v == p: continue
            dfs1(v, u, depth+1)
            W[u] += W[v]

    dfs1(1, 0, 0)

    # reroot dfs: compute f[u] for all u, track minimum
    ans = f1
    f = [0]*(N+1)
    f[1] = f1

    def dfs2(u, p):
        nonlocal ans
        for v in adj[u]:
            if v == p: continue
            # moving root from u to v:
            # distances to subtree v decrease by 1, to others increase by 1
            # so f[v] = f[u] + (S - 2*W[v])
            f[v] = f[u] + (S - 2*W[v])
            if f[v] < ans:
                ans = f[v]
            dfs2(v, u)

    dfs2(1, 0)

    print(ans)


if __name__ == "__main__":
    main()