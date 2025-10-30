class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        time = 0
        x = 1
        for s in strength:
            energy = 0
            curr_time = 0
            while energy < s:
                energy += x
                curr_time += 1
            time += curr_time
            x += K
        return time