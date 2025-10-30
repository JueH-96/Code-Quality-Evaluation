class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        max_coverage = 2000  # Since the maximum i is 1000, i+i = 2000
        next_dp = [0] * (max_coverage + 1)
        
        for i in range(n, 0, -1):
            current_dp = [float('inf')] * (max_coverage + 1)
            for coverage_end in range(max_coverage + 1):
                if i > coverage_end:
                    new_cov = max(coverage_end, i + i)
                    new_cov = min(new_cov, max_coverage)
                    current_dp[coverage_end] = prices[i-1] + next_dp[new_cov]
                else:
                    option1 = next_dp[coverage_end]
                    new_cov = max(coverage_end, i + i)
                    new_cov = min(new_cov, max_coverage)
                    option2 = prices[i-1] + next_dp[new_cov]
                    current_dp[coverage_end] = min(option1, option2)
            next_dp = current_dp.copy()
        
        return next_dp[0]