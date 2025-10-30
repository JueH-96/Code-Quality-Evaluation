class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        # Keep track of which baskets are available
        available = [True] * len(baskets)
        unplaced = 0
        
        # Process each fruit in order
        for fruit_qty in fruits:
            placed = False
            
            # Find leftmost available basket with sufficient capacity
            for i in range(len(baskets)):
                if available[i] and baskets[i] >= fruit_qty:
                    # Place fruit in this basket
                    available[i] = False
                    placed = True
                    break
            
            # If no suitable basket found, increment unplaced count
            if not placed:
                unplaced += 1
        
        return unplaced