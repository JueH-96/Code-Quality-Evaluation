class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        time = 0
        X = 1
        for s in strength:
            while X < s:
                X += K
                time += 1
            time += 1  # Time to break the lock
            X = K + 1  # Reset energy to 0 and increase X by K
        return time