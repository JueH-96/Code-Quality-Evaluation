import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]  # 1-based indexing

    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))

    # Check if the graph is strongly connected
    def is_strongly_connected():
        for u in range(1, N+1):
            visited = set()
            stack = [u]
            visited.add(u)
            while stack:
                current = stack.pop()
                for (v, w) in adj[current]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            if len(visited) != N:
                return False
        return True

    if not is_strongly_connected():
        print("No")
        return

    # Proceed with dynamic programming
    INF = float('inf')
    mask_count = 1 << N
    dp = [[INF] * (N + 1) for _ in range(mask_count)]

    # Initialize the DP table
    for u in range(1, N+1):
        mask = 1 << (u-1)
        dp[mask][u] = 0

    # Sort masks by the number of set bits
    sorted_masks = sorted(range(mask_count), key=lambda x: bin(x).count('1'))

    for mask in sorted_masks:
        for u in range(1, N+1):
            if (mask & (1 << (u-1))) == 0:
                continue  # u is not in the current mask
            if dp[mask][u] == INF:
                continue
            for (v, w) in adj[u]:
                new_mask = mask | (1 << (v-1))
                if dp[new_mask][v] > dp[mask][u] + w:
                    dp[new_mask][v] = dp[mask][u] + w

    full_mask = (1 << N) - 1
    ans = min(dp[full_mask][u] for u in range(1, N+1))
    print(ans)

if __name__ == "__main__":
    main()