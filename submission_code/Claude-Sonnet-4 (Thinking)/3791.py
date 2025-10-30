class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        occupied = [False] * len(baskets)
        unplaced_count = 0
        
        for fruit in fruits:
            placed = False
            for i in range(len(baskets)):
                if not occupied[i] and baskets[i] >= fruit:
                    occupied[i] = True
                    placed = True
                    break
            if not placed:
                unplaced_count += 1
        
        return unplaced_count