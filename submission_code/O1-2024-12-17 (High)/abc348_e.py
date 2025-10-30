def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B = map(int, input().split())
        adj[A].append(B)
        adj[B].append(A)
    C_in = list(map(int, input().split()))

    # 1-based indexing for convenience
    C = [0] + C_in

    # Sum of all weights
    S = sum(C)

    # Arrays to store subtree sums of C and dp values
    subTreeSum = [0]*(N+1)
    dp = [0]*(N+1)

    # We will compute dp[1] by doing a DFS from node 1 and summing C[v]*depth(v)
    dp1_temp = 0

    def dfs1(v, parent, depth):
        nonlocal dp1_temp
        subTreeSum[v] = C[v]
        dp1_temp += C[v] * depth
        for nxt in adj[v]:
            if nxt == parent:
                continue
            dfs1(nxt, v, depth+1)
            subTreeSum[v] += subTreeSum[nxt]

    # First DFS to compute dp[1] and subtree sums
    dfs1(1, -1, 0)
    dp[1] = dp1_temp
    min_dp = dp[1]

    # Second DFS to re-root and compute dp for other vertices
    def dfs2(v, parent):
        nonlocal min_dp
        for nxt in adj[v]:
            if nxt == parent:
                continue
            dp[nxt] = dp[v] + (S - 2*subTreeSum[nxt])
            if dp[nxt] < min_dp:
                min_dp = dp[nxt]
            dfs2(nxt, v)

    dfs2(1, -1)

    print(min_dp)

# Do not forget to call the main function
if __name__ == "__main__":
    main()