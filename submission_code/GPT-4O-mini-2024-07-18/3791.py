from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        # Sort the baskets to facilitate the placement of fruits
        baskets.sort()
        
        unplaced_count = 0
        basket_index = 0
        n = len(baskets)
        
        for fruit in fruits:
            # Find the leftmost basket that can hold the current fruit
            while basket_index < n and baskets[basket_index] < fruit:
                basket_index += 1
            
            # If we found a suitable basket, place the fruit
            if basket_index < n:
                # Place the fruit in the basket
                basket_index += 1  # Move to the next basket for future fruits
            else:
                # No suitable basket found for this fruit
                unplaced_count += 1
        
        return unplaced_count