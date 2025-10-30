class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        # Cumulative counts for each direction
        cum_E = [0] * (n + 1)
        cum_W = [0] * (n + 1)
        cum_N = [0] * (n + 1)
        cum_S = [0] * (n + 1)
        
        for i in range(n):
            cum_E[i + 1] = cum_E[i] + (1 if s[i] == 'E' else 0)
            cum_W[i + 1] = cum_W[i] + (1 if s[i] == 'W' else 0)
            cum_N[i + 1] = cum_N[i] + (1 if s[i] == 'N' else 0)
            cum_S[i + 1] = cum_S[i] + (1 if s[i] == 'S' else 0)
        
        # Iterate over t from n downto 1 to find the largest t where min cost <= k
        for t in range(n, 0, -1):
            # Compute the minimum cost for prefix up to t
            cost1 = t - cum_E[t]  # Single 'E'
            cost2 = t - cum_W[t]  # Single 'W'
            cost3 = t - cum_N[t]  # Single 'N'
            cost4 = t - cum_S[t]  # Single 'S'
            cost5 = cum_W[t] + cum_S[t]  # Pair ('E', 'N')
            cost6 = cum_W[t] + cum_N[t]  # Pair ('E', 'S')
            cost7 = cum_E[t] + cum_S[t]  # Pair ('W', 'N')
            cost8 = cum_E[t] + cum_N[t]  # Pair ('W', 'S')
            min_cost = min(cost1, cost2, cost3, cost4, cost5, cost6, cost7, cost8)
            
            if min_cost <= k:
                return t
        
        # The loop should always find a t since for t=1, min_cost <= 0 <= k, but return safely
        return 1