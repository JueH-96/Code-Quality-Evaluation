class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(i):
            s = str(i * i)
            n = len(s)
            for j in range(1 << (n - 1)):
                cur = 0
                temp = ""
                for k in range(n):
                    temp += s[k]
                    if (j >> k) & 1 or k == n - 1:
                        cur += int(temp)
                        temp = ""
                if cur == i:
                    return True
            return False

        ans = 0
        for i in range(1, n + 1):
            if check(i):
                ans += i * i
        return ans