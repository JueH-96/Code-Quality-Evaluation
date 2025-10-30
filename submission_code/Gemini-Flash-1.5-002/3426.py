class Solution:
    def minimumChairs(self, s: str) -> int:
        chairs = 0
        occupied = 0
        for event in s:
            if event == 'E':
                occupied += 1
                chairs = max(chairs, occupied)
            else:
                occupied -= 1
        return chairs