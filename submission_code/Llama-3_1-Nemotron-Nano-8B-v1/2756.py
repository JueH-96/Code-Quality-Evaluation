from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        min_sum = float('inf')
        n = len(prices)
        for i in range(n):
            for j in range(i + 1, n):
                current_sum = prices[i] + prices[j]
                if current_sum <= money and current_sum < min_sum:
                    min_sum = current_sum
        return money - min_sum if min_sum != float('inf') else money