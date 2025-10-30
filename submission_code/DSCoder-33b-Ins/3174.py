class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        s1 = ''.join(str(i % 2) for i in range(n))
        s2 = ''.join(str((i + 1) % 2) for i in range(n))
        cnt1, cnt2 = 0, 0
        for i in range(n):
            if s[i] != s1[i]:
                cnt1 += 1
            if s[i] != s2[i]:
                cnt2 += 1
        return min(cnt1, cnt2)