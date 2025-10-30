from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        used = set()
        maximumHeight.sort()
        
        for i in range(n):
            if maximumHeight[i] in used:
                if i == 0 or maximumHeight[i] <= maximumHeight[i-1]:
                    return -1
                maximumHeight[i] = maximumHeight[i-1] + 1
            used.add(maximumHeight[i])
        
        return sum(maximumHeight)