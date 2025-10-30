class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        from collections import defaultdict
        n = len(s)
        count = defaultdict(int)
        res = 0
        left = 0
        for right in range(n):
            count[s[right]] += 1
            while len(count) > 2:
                count[s[left]] -= 1
                if count[s[left]] == 0:
                    del count[s[left]]
                left += 1
            res = max(res, right - left + 1)
        return res