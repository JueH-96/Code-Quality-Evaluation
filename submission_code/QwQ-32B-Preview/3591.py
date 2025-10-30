class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        import sys
        INF = float('inf')
        alpha = {chr(ord('a') + i): i for i in range(26)}
        
        # Initialize cost matrix
        cost = [[INF for _ in range(26)] for _ in range(26)]
        for u in range(26):
            cost[u][u] = 0
            v_next = (u + 1) % 26
            v_prev = (u - 1) % 26
            cost[u][v_next] = nextCost[u]
            cost[u][v_prev] = previousCost[u]
        
        # Floyd-Warshall algorithm
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if cost[i][k] != INF and cost[k][j] != INF:
                        cost[i][j] = min(cost[i][j], cost[i][k] + cost[k][j])
        
        # Calculate total shift distance
        total_distance = 0
        for char_s, char_t in zip(s, t):
            s_idx = alpha[char_s]
            t_idx = alpha[char_t]
            total_distance += cost[s_idx][t_idx]
        
        return total_distance