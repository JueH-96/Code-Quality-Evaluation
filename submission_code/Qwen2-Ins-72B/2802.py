class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(s, i, total):
            if i == len(s):
                return total == 0
            for j in range(i + 1, len(s) + 1):
                num = int(s[i:j])
                if check(s, j, total - num):
                    return True
            return False
        
        res = 0
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            if check(square_str, 0, i):
                res += square
        return res