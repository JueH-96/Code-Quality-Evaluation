from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        """
        For each character c, let k = count of c in s.
        One can show independently for each character that the maximum
        number of removals is floor((k-1)/2), so the minimal number
        of c's left is:
            if k is odd: 1
            if k is even: 2  (and of course k>=2)
        Summing over all characters gives the final minimal length.
        """
        cnt = Counter(s)
        ans = 0
        for f in cnt.values():
            # f > 0 always for keys in Counter
            if f % 2 == 1:
                ans += 1
            else:
                ans += 2
        return ans