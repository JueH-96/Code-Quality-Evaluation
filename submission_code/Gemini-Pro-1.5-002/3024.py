class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        mod = 10**9 + 7
        shifts = []
        for i in range(n):
            if s[i:] + s[:i] == t:
                shifts.append(i)
        
        if not shifts:
            return 0
        
        ans = 0
        for shift in shifts:
            if k == 1:
                ans += 1
            elif shift == 0:
                if k % n == 0:
                    ans += 1
            else:
                if (k - shift) % n == 0 and k >= shift:
                    ans += 1
        return ans % mod