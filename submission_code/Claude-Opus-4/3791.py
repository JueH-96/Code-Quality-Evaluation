class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        used = [False] * n  # Track which baskets have been used
        unplaced = 0
        
        for i in range(n):
            fruit_quantity = fruits[i]
            placed = False
            
            # Try to find the leftmost available basket with sufficient capacity
            for j in range(n):
                if not used[j] and baskets[j] >= fruit_quantity:
                    used[j] = True
                    placed = True
                    break
            
            if not placed:
                unplaced += 1
        
        return unplaced