import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N, M = map(int, input().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    color = [ -1 ] * (N+1)
    sum_ab_mod2 = 0
    sum_sizes_sq = 0
    from collections import deque
    for i in range(1, N+1):
        if color[i] != -1: continue
        # new component
        q = deque([i])
        color[i] = 0
        cnt0 = 1
        cnt1 = 0
        while q:
            u = q.popleft()
            for w in adj[u]:
                if color[w] == -1:
                    color[w] = color[u]^1
                    if color[w] == 0: cnt0 += 1
                    else: cnt1 += 1
                    q.append(w)
        # in this comp, bipartite edges possible = cnt0*cnt1
        sum_ab_mod2 ^= ((cnt0 & 1) & (cnt1 & 1))
        s = cnt0 + cnt1
        sum_sizes_sq += s*s
    # t1 = (sum over comps ab mod2) - M mod2
    t1 = (sum_ab_mod2 - (M & 1)) & 1
    # t2 = ((N^2 - sum_sizes_sq)//2) mod2
    diff = N*N - sum_sizes_sq
    t2 = ((diff // 2) & 1)
    res = (t1 + t2) & 1
    if res == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == "__main__":
    main()