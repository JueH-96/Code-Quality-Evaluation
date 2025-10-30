class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        # Keep track of which baskets are used
        used_baskets = [False] * n
        unplaced = 0
        
        # For each fruit type
        for i in range(n):
            fruit_quantity = fruits[i]
            placed = False
            
            # Try to place in leftmost available basket with sufficient capacity
            for j in range(n):
                if not used_baskets[j] and baskets[j] >= fruit_quantity:
                    used_baskets[j] = True
                    placed = True
                    break
            
            # If fruit couldn't be placed in any basket, increment unplaced counter
            if not placed:
                unplaced += 1
                
        return unplaced