class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        m = n // 2
        INF = float('inf')
        
        # Initialize DP for the first pair (i=0)
        prev_dp = [[INF] * 3 for _ in range(3)]
        first_left = 0
        first_right = n - 1
        for a in range(3):
            for b in range(3):
                if a != b:
                    prev_dp[a][b] = cost[first_left][a] + cost[first_right][b]
        
        for i in range(1, m):
            current_left = i
            current_right = n - 1 - i
            new_dp = [[INF] * 3 for _ in range(3)]
            for a_prev in range(3):
                for b_prev in range(3):
                    if prev_dp[a_prev][b_prev] == INF:
                        continue
                    # Try all possible current left (c) and right (d) colors
                    for c in range(3):
                        if c == a_prev:
                            continue  # current left must differ from previous left
                        for d in range(3):
                            if c == d:
                                continue  # current left and right must differ
                            if d == b_prev:
                                continue  # current right must differ from previous right
                            # Update new_dp with the new state (c, d)
                            new_cost = prev_dp[a_prev][b_prev] + cost[current_left][c] + cost[current_right][d]
                            if new_cost < new_dp[c][d]:
                                new_dp[c][d] = new_cost
            prev_dp = new_dp
        
        # Find the minimum cost among all possible states after processing all pairs
        min_total = INF
        for a in range(3):
            for b in range(3):
                if prev_dp[a][b] < min_total:
                    min_total = prev_dp[a][b]
        return min_total