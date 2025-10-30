class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        total_cost = 0
        free_until = -1  # Initialize to -1 since positions are 0-indexed
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + prices[i]

        for i in range(n):
            if i <= free_until:
                # Additional benefit if we purchase fruit i
                benefit_start = max(free_until + 1, i + 1)
                benefit_end = min(n - 1, i + i)
                if benefit_start > benefit_end:
                    additional_benefit = 0
                else:
                    additional_benefit = prefix_sum[benefit_end + 1] - prefix_sum[benefit_start]
                if additional_benefit >= prices[i]:
                    total_cost += prices[i]
                    free_until = max(free_until, i + i)
            else:
                # Must purchase fruit i
                total_cost += prices[i]
                free_until = max(free_until, i + i)
        return total_cost