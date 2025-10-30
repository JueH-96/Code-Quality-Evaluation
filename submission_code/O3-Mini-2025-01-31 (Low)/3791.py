from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        unplaced = 0
        basket_index = 0
        
        # Process each fruit in the given order
        for fruit in fruits:
            # Find the leftmost available basket that can accommodate the fruit
            while basket_index < n and baskets[basket_index] < fruit:
                basket_index += 1
            # If no basket is found, count this fruit as unplaced
            if basket_index == n:
                unplaced += 1
            else:
                # Place the fruit in this basket and move to next basket
                basket_index += 1
                
        return unplaced