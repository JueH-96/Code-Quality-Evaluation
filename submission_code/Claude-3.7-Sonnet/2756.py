class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Initialize the two minimum prices with infinity
        min1 = float('inf')
        min2 = float('inf')
        
        # Find the two minimum prices in one pass
        for price in prices:
            if price < min1:
                min2 = min1
                min1 = price
            elif price < min2:
                min2 = price
        
        # Calculate the total cost of the two cheapest chocolates
        total_cost = min1 + min2
        
        # Check if we can afford both chocolates
        if total_cost <= money:
            return money - total_cost  # Return leftover money
        else:
            return money  # Can't buy two chocolates, return original money