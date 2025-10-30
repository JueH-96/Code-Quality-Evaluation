class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        total = 0
        for x in possible:
            total += 1 if x == 1 else -1
        
        current = 0
        n = len(possible)
        for i in range(n - 1):
            current += 1 if possible[i] == 1 else -1
            if 2 * current > total:
                return i + 1
        return -1