class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        cnt, last = Counter(s), []
        for c in reversed(s):
            if cnt[c] > 1:
                last.append(c)
                cnt[c] -= 1
        return ''.join(last)