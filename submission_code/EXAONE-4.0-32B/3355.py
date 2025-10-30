class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        total = 0
        for x in possible:
            if x == 1:
                total += 1
            else:
                total -= 1
        
        cur = 0
        n = len(possible)
        for i in range(n - 1):
            if possible[i] == 1:
                cur += 1
            else:
                cur -= 1
            if 2 * cur > total:
                return i + 1
        
        return -1