class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        import sys
        INF = sys.maxsize
        NUM_LETTERS = 26

        # Initialize adjacency matrix
        adj = [[INF for _ in range(NUM_LETTERS)] for _ in range(NUM_LETTERS)]
        for i in range(NUM_LETTERS):
            adj[i][i] = 0
            next_idx = (i + 1) % NUM_LETTERS
            adj[i][next_idx] = nextCost[i]
            prev_idx = (i - 1 + NUM_LETTERS) % NUM_LETTERS
            adj[i][prev_idx] = previousCost[i]

        # Floyd-Warshall algorithm to find shortest paths between all pairs
        for k in range(NUM_LETTERS):
            for i in range(NUM_LETTERS):
                for j in range(NUM_LETTERS):
                    if adj[i][k] != INF and adj[k][j] != INF:
                        if adj[i][j] > adj[i][k] + adj[k][j]:
                            adj[i][j] = adj[i][k] + adj[k][j]

        # Calculate total shift distance
        total_cost = 0
        for sc, tc in zip(s, t):
            a = ord(sc) - ord('a')
            b = ord(tc) - ord('a')
            total_cost += adj[a][b]

        return total_cost