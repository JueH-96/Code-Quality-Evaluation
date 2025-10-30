class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to try the cheapest combinations first
        prices.sort()
        
        # Initialize the minimum sum to be greater than the maximum possible money
        min_sum = float('inf')
        
        # Try every combination of two chocolates
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                # Calculate the sum of two chocolates
                sum_of_two = prices[i] + prices[j]
                # If the sum is less than or equal to the money we have
                # and less than the current minimum sum, update the minimum sum
                if sum_of_two <= money and sum_of_two < min_sum:
                    min_sum = sum_of_two
        
        # If we found a valid combination, return the leftover money
        # Otherwise, return the initial amount of money
        return money - min_sum if min_sum != float('inf') else money