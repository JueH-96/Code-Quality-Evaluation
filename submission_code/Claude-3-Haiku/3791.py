class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        baskets.sort(reverse=True)
        
        unplaced = 0
        i = 0
        for fruit in fruits:
            while i < n and baskets[i] < fruit:
                i += 1
            if i == n:
                unplaced += 1
            else:
                i += 1
        
        return unplaced