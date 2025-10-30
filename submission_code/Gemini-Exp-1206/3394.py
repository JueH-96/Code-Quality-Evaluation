class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        ans = x
        cnt = 0
        for i in range(32):
            if (x >> i) & 1 == 0:
                if cnt < n - 1:
                    ans |= (1 << i)
                    cnt += 1
        if cnt < n - 1:
            ans |= (1 << 31)
            for i in range(31):
                if (ans >> i) & 1 == 0:
                    if cnt < n - 1:
                        ans |= (1 << i)
                        cnt += 1
        return ans