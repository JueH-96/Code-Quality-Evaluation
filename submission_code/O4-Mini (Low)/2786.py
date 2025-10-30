class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        # isPair[i] == 1 if s[i] == s[i-1], else 0 (for i>=1)
        isPair = [0]*n
        for i in range(1, n):
            if s[i] == s[i-1]:
                isPair[i] = 1

        l = 0
        pairCount = 0
        res = 1

        for r in range(n):
            # include r
            if isPair[r]:
                pairCount += 1
            # shrink from left while more than one pair in window
            while pairCount > 1:
                # if the pair at index l+1 was counted, remove it
                if isPair[l+1]:
                    pairCount -= 1
                l += 1
            # update result
            res = max(res, r - l + 1)

        return res