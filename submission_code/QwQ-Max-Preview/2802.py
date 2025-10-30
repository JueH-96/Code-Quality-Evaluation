class Solution:
    def punishmentNumber(self, n: int) -> int:
        def helper(s, start, current_sum, target):
            if start == len(s):
                return current_sum == target
            for end in range(start + 1, len(s) + 1):
                part = int(s[start:end])
                if helper(s, end, current_sum + part, target):
                    return True
            return False
        
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            if helper(s, 0, 0, i):
                total += square
        return total