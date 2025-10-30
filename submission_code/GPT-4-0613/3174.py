class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        cnt = [0, 0]  # count of '0' and '1'
        for i in range(n):
            cnt[int(s[i])] += 1

        ans = min(cnt)  # minimum number of changes
        cnt = [0, 0]  # reset count of '0' and '1'
        for i in range(n):
            cnt[int(s[i])] += 1
            if i % 2 == 1:  # check every two characters
                ans = min(ans, min(cnt[0] + (n // 2 - cnt[1]), cnt[1] + (n // 2 - cnt[0])))
        return ans