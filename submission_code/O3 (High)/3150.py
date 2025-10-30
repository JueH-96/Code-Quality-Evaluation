class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        """
        Returns the lexicographically smallest substring of `s`
        that contains exactly `k` ones and whose length is minimal
        among all such substrings.
        """
        if k > s.count('1'):            # impossible to have k ones
            return ""
        
        n = len(s)
        min_len = n + 1                 # current best (shortest) length
        best   = ""                     # current best (lexicographically smallest) substring
        
        for left in range(n):
            ones = 0
            for right in range(left, n):
                if s[right] == '1':
                    ones += 1
                if ones > k:            # too many 1s, cannot recover by extending
                    break
                if ones == k:
                    cur_len = right - left + 1
                    if cur_len < min_len:
                        min_len = cur_len
                        best = s[left:right+1]
                    elif cur_len == min_len:
                        cand = s[left:right+1]
                        if best == "" or cand < best:
                            best = cand
                    break               # any longer substring from `left`
                                        # will be strictly longer, so stop
        return best