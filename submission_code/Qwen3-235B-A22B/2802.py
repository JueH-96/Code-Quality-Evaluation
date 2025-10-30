class Solution:
    def punishmentNumber(self, n: int) -> int:
        res = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            if self.check(s, i):
                res += square
        return res
    
    def check(self, s, target):
        return self.dfs(s, target, 0, 0)
    
    def dfs(self, s, target, pos, current_sum):
        if pos == len(s):
            return current_sum == target
        if current_sum > target:
            return False
        for end in range(pos + 1, len(s) + 1):
            num = int(s[pos:end])
            if self.dfs(s, target, end, current_sum + num):
                return True
        return False