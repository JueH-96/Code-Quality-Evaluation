class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        # Sort fruits and baskets to facilitate the allocation process
        fruits.sort()
        baskets.sort()
        
        # Initialize pointers for fruits and baskets
        fruit_index = 0
        basket_index = 0
        n = len(fruits)
        
        # Iterate over both lists to place fruits in baskets
        while fruit_index < n and basket_index < n:
            if fruits[fruit_index] <= baskets[basket_index]:
                # If the current fruit can be placed in the current basket
                fruit_index += 1  # Move to the next fruit
            # Move to the next basket in either case
            basket_index += 1
        
        # The number of unplaced fruits is the remaining fruits that couldn't be placed
        return n - fruit_index