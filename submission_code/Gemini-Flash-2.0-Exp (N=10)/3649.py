class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        time = 0
        energy = 0
        factor = 1
        
        for s in strength:
            while energy < s:
                time += 1
                energy += factor
            energy = 0
            factor += K
        
        return time