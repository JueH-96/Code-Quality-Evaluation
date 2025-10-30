class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers by the end index
        offers.sort(key=lambda x: x[1])

        # Initialize a list to store the maximum gold that can be earned up to each house
        dp = [0] * n

        # Dictionary to store the maximum gold for each end index
        max_gold = {}

        for offer in offers:
            start, end, gold = offer
            # Calculate the maximum gold that can be earned up to the start index
            max_gold[end] = max(max_gold.get(end, 0), gold + (dp[start - 1] if start > 0 else 0))

        # Fill the dp array with the maximum gold that can be earned up to each house
        for i in range(n):
            dp[i] = max(dp[i - 1] if i > 0 else 0, max_gold.get(i, 0))

        return dp[-1]