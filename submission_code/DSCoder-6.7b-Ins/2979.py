class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers by end point
        offers.sort(key=lambda x: x[1])

        # Initialize dp array with 0s
        dp = [0] * (n + 1)

        # Initialize max_gold variable to keep track of maximum gold earned
        max_gold = 0

        # Iterate over the offers
        for start, end, gold in offers:
            # Update max_gold with the maximum of current max_gold and dp[start] + gold
            max_gold = max(max_gold, dp[start] + gold)
            # Update dp[end + 1] with the maximum of current dp[end + 1] and max_gold
            if end + 1 <= n:
                dp[end + 1] = max(dp[end + 1], max_gold)

        # Return the maximum gold earned
        return max_gold