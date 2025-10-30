from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # We need to pick a subset S of walls for the paid painter
        # so that sum_{i in S}(time[i] + 1) >= n,
        # and we minimize sum_{i in S} cost[i].
        # This is a 0/1 knapsack where each item i has
        # weight w_i = time[i] + 1, value cost[i],
        # and we want minimum total cost to reach total weight >= n.
        
        # dp[w] = minimum cost to achieve total weight exactly w (capped at n)
        INF = 10**18
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            w = time[i] + 1
            c = cost[i]
            # traverse backwards to avoid reuse
            for cur_w in range(n, -1, -1):
                if dp[cur_w] == INF:
                    continue
                new_w = cur_w + w
                if new_w >= n:
                    new_w = n
                if dp[new_w] > dp[cur_w] + c:
                    dp[new_w] = dp[cur_w] + c
        
        return dp[n]

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.paintWalls([1,2,3,2], [1,2,3,2]))  # Output: 3
    print(sol.paintWalls([2,3,4,2], [1,1,1,1]))  # Output: 4