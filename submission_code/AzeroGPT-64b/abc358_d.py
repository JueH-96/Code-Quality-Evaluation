from typing import List
from bisect import bisect_left, insort_left

class Solution:
    def buyCandy(self, n: int, m: int, prices: List[int], min_candies: List[int]) -> int:
        # Pair each candy box with its price and the minimum number of candies it contains
        pairs = list(zip(prices, min_candies))
        # Sort the pairs based on the minimum number of candies in descending order
        pairs.sort(key=lambda x: -x[1])
        min_candies_sorted = [pair[1] for pair in pairs]
        prices_sorted = [pair[0] for pair in pairs]
        
        # Convert the sorted list of prices to a sorted set for efficient removal
        price_list = sorted(prices_sorted)
        extra_candies = 0
        extra_cost = 0
        
        # Find the total amount of extra candies
        for i in range(m):
            extra_candies += max(0, min_candies[i] - min_candies_sorted[i])
        
        if extra_candies > sum(min_candies_sorted[m:]):
            return -1
        
        # Find the minimum possible total price with necessary candies
        mp = sum(price for candy, price in pairs[:m])
        ans = mp
        min_candies_sorted += [0] * m
        
        for i in range(n):
            candy, price = pairs[i]
            # Find the number of candies in the current box that are not strictly necessary
            extra_candies -= max(0, min_candies[i] - min_candies_sorted[i])
            min_candies_sorted[i] = candy
            insort_left(min_candies_sorted, candy, hi=m)
            extra_candies += max(0, min_candies_sorted[i] - min_candies_sorted[i + m])
            
            if extra_candies > 0:  # Find the index of the first box that contains extra candies
                pos = bisect_left(min_candies_sorted, extra_candies, hi=m)
                ans = min(ans, mp + price_list[pos] - price)
            
            mp += price
            insort_left(price_list, price)
        
        return ans