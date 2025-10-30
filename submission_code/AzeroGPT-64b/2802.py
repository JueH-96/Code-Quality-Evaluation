class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(s, idx, k):
            if idx == len(s) and k == 0:
                return True
            for i in range(idx + 1, len(s) + 1):
                num = int(s[idx:i])
                if num <= k and check(s, i, k-num):
                    return True
            return False
        
        ans = 0
        for i in range(1, n + 1):
            if check(f'{i * i}', 0, i):
                ans += i*i
        return ans