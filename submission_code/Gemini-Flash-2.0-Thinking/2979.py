class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers_by_end = {}
        for start, end, gold in offers:
            if end not in offers_by_end:
                offers_by_end[end] = []
            offers_by_end[end].append((start, gold))

        dp = [0] * (n + 1)  # dp[i] is the max profit considering houses 0 to i-1

        for i in range(1, n + 1):
            dp[i] = dp[i - 1]  # Option: don't sell house i-1

            if i - 1 in offers_by_end:
                for start, gold in offers_by_end[i - 1]:
                    dp[i] = max(dp[i], gold + dp[start])

        return dp[n]