class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Build the adjacency matrix
        INF = float('inf')
        dist = [[INF]*26 for _ in range(26)]
        for c in range(26):
            dist[c][c] = 0  # Distance to itself is 0

        for c in range(26):
            next_c = (c + 1) % 26
            prev_c = (c - 1 + 26) % 26
            # Edge from c to next_c with cost nextCost[c]
            dist[c][next_c] = min(dist[c][next_c], nextCost[c])
            # Edge from c to prev_c with cost previousCost[c]
            dist[c][prev_c] = min(dist[c][prev_c], previousCost[c])

        # Run Floyd-Warshall Algorithm
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        total_cost = 0
        for i in range(len(s)):
            c1 = ord(s[i]) - ord('a')
            c2 = ord(t[i]) - ord('a')
            total_cost += dist[c1][c2]

        return total_cost