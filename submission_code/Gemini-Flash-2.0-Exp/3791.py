class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        placed = [False] * n
        
        for i in range(n):
            placed_fruit = False
            for j in range(n):
                if not placed[j] and baskets[j] >= fruits[i]:
                    placed[j] = True
                    placed_fruit = True
                    break
            
        unplaced_count = 0
        for p in placed:
            if not p:
                unplaced_count += 1
        
        return len(fruits) - n + unplaced_count