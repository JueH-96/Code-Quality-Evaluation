class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        # Sort fruits and baskets to facilitate the placement
        fruits.sort()
        baskets.sort()
        
        # Initialize pointers for fruits and baskets
        fruit_index = 0
        basket_index = 0
        unplaced_fruits = 0
        
        # Number of fruits
        n = len(fruits)
        
        # Try to place each fruit in the appropriate basket
        while fruit_index < n:
            # If there are no more baskets to check, count all remaining fruits as unplaced
            if basket_index >= n:
                unplaced_fruits += (n - fruit_index)
                break
            
            # Check if the current fruit can be placed in the current basket
            if fruits[fruit_index] <= baskets[basket_index]:
                # Move to the next fruit and the next basket
                fruit_index += 1
                basket_index += 1
            else:
                # Current basket cannot hold the fruit, move to the next basket
                basket_index += 1
        
        # If some fruits are left without checking baskets, they are unplaced
        if fruit_index < n:
            unplaced_fruits += (n - fruit_index)
        
        return unplaced_fruits