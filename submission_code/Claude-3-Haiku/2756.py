class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        min_sum = float('inf')
        leftover = money
        
        for i in range(len(prices)):
            for j in range(i+1, len(prices)):
                total = prices[i] + prices[j]
                if total <= money:
                    leftover = money - total
                    min_sum = min(min_sum, leftover)
        
        return min_sum if min_sum != float('inf') else money