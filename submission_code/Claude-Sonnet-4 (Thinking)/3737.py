class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(pos, left_color, right_color):
            if pos == n // 2:
                return 0
            
            i, j = pos, n - 1 - pos
            min_cost = float('inf')
            
            for color_i in range(3):
                for color_j in range(3):
                    if color_i == color_j:
                        continue  # Equidistant constraint
                    
                    # Adjacent constraint for house i
                    if i > 0 and color_i == left_color:
                        continue
                    
                    # Adjacent constraint for house j
                    if j < n - 1 and color_j == right_color:
                        continue
                    
                    current_cost = cost[i][color_i] + cost[j][color_j]
                    min_cost = min(min_cost, current_cost + dp(pos + 1, color_i, color_j))
            
            return min_cost
        
        return dp(0, -1, -1)