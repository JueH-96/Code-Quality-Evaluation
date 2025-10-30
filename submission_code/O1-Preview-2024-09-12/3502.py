class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        ans = 0
        for i in range(n):
            cnt = {}
            for j in range(i, n):
                cnt[s[j]] = cnt.get(s[j], 0) + 1
                if any(v >= k for v in cnt.values()):
                    ans += 1
        return ans