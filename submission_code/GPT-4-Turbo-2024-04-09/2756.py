class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to easily find the minimum combinations
        prices.sort()
        
        # Initialize the minimum leftover as the initial money (no purchase case)
        min_leftover = money
        
        # Try to find the two least expensive chocolates that can be bought
        n = len(prices)
        for i in range(n):
            for j in range(i + 1, n):
                total_cost = prices[i] + prices[j]
                if total_cost <= money:
                    # Calculate leftover money after buying these two chocolates
                    leftover = money - total_cost
                    # We want to minimize the leftover to spend as much as possible
                    # but it should be non-negative
                    if leftover < min_leftover:
                        min_leftover = leftover
                else:
                    # Since the list is sorted, all further combinations will be more expensive
                    break
        
        # If no valid purchase was made, return the initial money
        return min_leftover