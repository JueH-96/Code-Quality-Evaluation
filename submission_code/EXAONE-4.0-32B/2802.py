class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        for i in range(1, n + 1):
            sq = i * i
            s = str(sq)
            if self.check_partition(s, i):
                total += sq
        return total

    def check_partition(self, s, target):
        n_len = len(s)
        
        def dfs(start, t):
            if start == n_len:
                return t == 0
            if t < 0:
                return False
            num = 0
            for j in range(start, n_len):
                num = num * 10 + int(s[j])
                if num > t:
                    break
                if dfs(j + 1, t - num):
                    return True
            return False
        
        return dfs(0, target)