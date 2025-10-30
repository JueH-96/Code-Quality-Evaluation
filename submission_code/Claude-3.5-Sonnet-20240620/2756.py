class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        min1, min2 = float('inf'), float('inf')
        
        for price in prices:
            if price < min1:
                min2 = min1
                min1 = price
            elif price < min2:
                min2 = price
        
        total_cost = min1 + min2
        
        if total_cost <= money:
            return money - total_cost
        else:
            return money