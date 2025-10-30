class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_split(s, target):
            if target < 0:
                return False
            if len(s) == 0:
                return target == 0
            for i in range(1, len(s) + 1):
                first = s[:i]
                val = int(first)
                if can_split(s[i:], target - val):
                    return True
            return False
        
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            if can_split(s, i):
                total += square
        return total