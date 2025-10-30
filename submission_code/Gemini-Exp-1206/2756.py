class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        min_price = float('inf')
        second_min_price = float('inf')
        for price in prices:
            if price < min_price:
                second_min_price = min_price
                min_price = price
            elif price < second_min_price:
                second_min_price = price
        
        if min_price + second_min_price <= money:
            return money - (min_price + second_min_price)
        else:
            return money