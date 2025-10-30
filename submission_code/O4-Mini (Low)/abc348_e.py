import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(3000000)
    input = sys.stdin.readline

    N = int(input().strip())
    edges = [[] for _ in range(N)]
    for _ in range(N-1):
        a,b = map(int, input().split())
        a -= 1; b -= 1
        edges[a].append(b)
        edges[b].append(a)
    C = list(map(int, input().split()))

    total_weight = sum(C)
    subtree_weight = [0]*N
    # f_root will hold f(0) = sum_{i} C[i] * dist(0,i)
    f_root = 0

    # First DFS to compute subtree_weight and f_root
    def dfs1(u, p, depth):
        nonlocal f_root
        sw = C[u]
        f_root += C[u] * depth
        for v in edges[u]:
            if v == p: continue
            sw_v = dfs1(v, u, depth+1)
            sw += sw_v
        subtree_weight[u] = sw
        return sw

    dfs1(0, -1, 0)

    # Now reroot to compute f(v) for all v
    ans = f_root
    f = [0]*N
    f[0] = f_root

    def dfs2(u, p):
        nonlocal ans
        for v in edges[u]:
            if v == p: continue
            # when we move root from u to v:
            # distances to v's subtree decrease by 1 (sum C in subtree_weight[v])
            # distances to all others increase by 1 (total_weight - subtree_weight[v])
            # so f[v] = f[u] - subtree_weight[v] + (total_weight - subtree_weight[v])
            f[v] = f[u] + total_weight - 2*subtree_weight[v]
            if f[v] < ans:
                ans = f[v]
            dfs2(v, u)

    dfs2(0, -1)

    print(ans)


if __name__ == "__main__":
    main()