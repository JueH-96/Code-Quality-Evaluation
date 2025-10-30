from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        used = [False] * n
        min_unused = 0
        unplaced = 0

        for qty in fruits:
            i = min_unused
            while i < n and (used[i] or baskets[i] < qty):
                i += 1
            if i < n and not used[i]:
                used[i] = True
            else:
                unplaced += 1
            # Set min_unused to the first unused basket index
            while min_unused < n and used[min_unused]:
                min_unused += 1

        return unplaced