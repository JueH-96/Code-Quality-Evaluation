# YOUR CODE HERE

import sys
import threading

sys.setrecursionlimit(1 << 25)

def main():
    import sys
    N = int(sys.stdin.readline())
    tree = [[] for _ in range(N)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -=1
        v -=1
        tree[u].append(v)
        tree[v].append(u)

    max_dp = 1
    visited = [False] * N
    dp = [1]*N
    height = [0]*N

    def dfs(u, parent):
        nonlocal max_dp
        if len(tree[u]) == 1 and parent != -1:
            dp[u] = 1
            height[u] = 0
            return
        child_heights = []
        dp_u = 1
        max_child_height = 0
        for v in tree[u]:
            if v != parent:
                dfs(v, u)
                if height[v] < 2:
                    dp_u += dp[v]
                    max_child_height = max(max_child_height, height[v])
        height[u] = max_child_height + 1
        if height[u] > 2:
            dp[u] = 1
            height[u] = 0
        else:
            dp[u] = dp_u
            max_dp = max(max_dp, dp[u])

    for u in range(N):
        if len(tree[u]) > 1:
            dfs(u, -1)
            max_dp = max(max_dp, dp[u])

    print(N - max_dp)

threading.Thread(target=main).start()