class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(s, num):
            if not s and num == 0:
                return True
            if not s:
                return False

            for i in range(1, len(s) + 1):
                val = int(s[:i])
                if val > num:
                    break
                if check(s[i:], num - val):
                    return True
            return False

        ans = 0
        for i in range(1, n + 1):
            sq = i * i
            if check(str(sq), i):
                ans += sq
        return ans