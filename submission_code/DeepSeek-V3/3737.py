class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        if n == 0:
            return 0
        
        total_pairs = n // 2
        # For each pair, we'll track the best costs for each possible color combination (a, b) where a != b
        # Initialize for the first pair (0, n-1)
        # The DP will track for each possible (a, b) the minimal cost up to that pair
        
        # Initialize DP for the first pair
        # There are 6 possible (a, b) pairs where a and b are 0,1,2 and a != b
        prev_dp = {}
        colors = [0, 1, 2]
        for a in colors:
            for b in colors:
                if a != b:
                    prev_dp[(a, b)] = cost[0][a] + cost[n-1][b]
        
        for i in range(1, total_pairs):
            current_dp = {}
            left_house = i
            right_house = n - 1 - i
            for a in colors:
                for b in colors:
                    if a != b:
                        min_prev_cost = float('inf')
                        # The previous pair's colors (prev_a, prev_b) must satisfy:
                        # prev_b != a (since the right_house of previous pair is adjacent to left_house of current)
                        for prev_a, prev_b in prev_dp:
                            if prev_b != a:
                                current_cost = prev_dp[(prev_a, prev_b)] + cost[left_house][a] + cost[right_house][b]
                                if current_cost < min_prev_cost:
                                    min_prev_cost = current_cost
                        current_dp[(a, b)] = min_prev_cost
            prev_dp = current_dp
        
        # The minimal value among all possible pairs in the last step
        return min(prev_dp.values())