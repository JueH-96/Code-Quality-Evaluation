from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Find the two smallest distinct prices in one pass
        smallest = second_smallest = float('inf')
        
        for price in prices:
            if price < smallest:
                second_smallest = smallest
                smallest = price
            elif price < second_smallest:
                second_smallest = price
        
        total_cost = smallest + second_smallest
        
        # If we can afford the two cheapest chocolates, return leftover
        if total_cost <= money:
            return money - total_cost
        
        # Otherwise, buying two chocolates is impossible without debt
        return money