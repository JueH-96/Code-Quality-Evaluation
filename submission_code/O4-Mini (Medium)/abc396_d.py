import sys
sys.setrecursionlimit(10**7)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    # build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        adj[u].append((v, w))
        adj[v].append((u, w))

    # ans[0] will hold the minimum xor found
    # initialize to a large value
    ans = [1 << 64]

    visited = [False] * (N+1)

    def dfs(u, curr_xor):
        # if we've reached N, update answer
        if u == N:
            if curr_xor < ans[0]:
                ans[0] = curr_xor
            return

        # continue DFS
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                dfs(v, curr_xor ^ w)
                visited[v] = False

    # start DFS from 1
    visited[1] = True
    dfs(1, 0)

    print(ans[0])

if __name__ == "__main__":
    main()