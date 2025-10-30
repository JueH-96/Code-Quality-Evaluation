class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        unplaced = 0
        
        # Track which baskets are already used
        used_baskets = [False] * n
        
        for i in range(n):
            fruit_quantity = fruits[i]
            placed = False
            
            # Find the leftmost available basket with sufficient capacity
            for j in range(n):
                if not used_baskets[j] and baskets[j] >= fruit_quantity:
                    used_baskets[j] = True
                    placed = True
                    break
            
            # If no suitable basket was found, this fruit remains unplaced
            if not placed:
                unplaced += 1
        
        return unplaced