from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        Calculates the minimum cost to reach each position in a line by finding the prefix minimum of the cost array.

        The minimum cost to reach position `i` is determined by the cheapest single paid swap that allows us to land at or before `i`.
        If we pay `cost[j]` (where j <= i), we land at position `j`. From there, we can move to any position `k > j` for free.
        Therefore, to reach `i`, we can pay `cost[j]` for any `j <= i` and then move to `i` for free.
        To minimize the cost, we should choose the `j <= i` with the minimum `cost[j]`.
        
        This means `answer[i] = min(cost[0], cost[1], ..., cost[i])`.
        This can be computed efficiently in O(n) time.
        """
        n = len(cost)
        if n == 0:
            return []
        
        # Initialize an array to store the results.
        answer = [0] * n
        
        # The cost to reach position 0 is just cost[0], as it's the only option.
        answer[0] = cost[0]
        
        # For each subsequent position i, the minimum cost is the minimum of:
        # 1. The minimum cost to reach the previous position (i-1), which is min(cost[0]...cost[i-1]).
        # 2. The cost to jump directly to position i.
        for i in range(1, n):
            answer[i] = min(answer[i-1], cost[i])
            
        return answer