from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers by the starting index
        offers.sort(key=lambda x: x[0])

        # Initialize the dp array with zeros
        dp = [0] * (n + 1)

        # Iterate through each offer
        for start, end, gold in offers:
            # The maximum gold we can get by including this offer
            max_gold = dp[start] + gold
            # Update the dp values for the range [end + 1, n]
            for i in range(end + 1, n + 1):
                dp[i] = max(dp[i], max_gold)

        # The answer is the maximum value in the dp array
        return max(dp)

# Example usage:
# sol = Solution()
# print(sol.maximizeTheProfit(5, [[0,0,1],[0,2,2],[1,3,2]]))  # Output: 3
# print(sol.maximizeTheProfit(5, [[0,0,1],[0,2,10],[1,3,2]]))  # Output: 10