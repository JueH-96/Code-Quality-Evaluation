class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Find the two minimum prices
        min1 = float('inf')
        min2 = float('inf')
        
        for price in prices:
            if price < min1:
                min2 = min1
                min1 = price
            elif price < min2:
                min2 = price
        
        # Check if we can afford the two cheapest chocolates
        min_cost = min1 + min2
        
        if min_cost <= money:
            return money - min_cost
        else:
            return money