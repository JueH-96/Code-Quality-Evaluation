import bisect

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        baskets_sorted = sorted(baskets)
        used = [False] * len(baskets_sorted)
        unplaced = 0
        
        for fruit in fruits:
            # Find the leftmost basket >= fruit that is not used
            index = bisect.bisect_left(baskets_sorted, fruit)
            while index < len(baskets_sorted):
                if not used[index]:
                    used[index] = True
                    break
                index += 1
            else:
                unplaced += 1
        return unplaced