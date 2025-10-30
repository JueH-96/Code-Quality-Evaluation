class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            if self.can_split(s, i):
                total += square
        return total
    
    def can_split(self, s: str, target: int) -> bool:
        def helper(index, current_sum):
            if index == len(s):
                return current_sum == target
            for i in range(index, len(s)):
                num = int(s[index:i+1])
                if current_sum + num > target:
                    break
                if helper(i + 1, current_sum + num):
                    return True
            return False
        return helper(0, 0)