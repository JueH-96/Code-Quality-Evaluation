class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices in ascending order
        sorted_prices = sorted(prices)
        # Sum of the two smallest prices
        sum_two_smallest = sorted_prices[0] + sorted_prices[1]
        # Check if the sum is within the budget
        if sum_two_smallest <= money:
            return money - sum_two_smallest
        else:
            return money