from bisect import bisect_left

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # Record positions of each character
        pos = [[] for _ in range(26)]
        for i, ch in enumerate(s):
            pos[ord(ch) - ord('a')].append(i)
        
        ans = 0
        INF = n + 1
        
        # For each start index i
        for i in range(n):
            # Find, for each character, the position of its k-th occurrence ≥ i
            min_j = INF
            for c in range(26):
                lst = pos[c]
                # find first occurrence index in lst ≥ i
                idx = bisect_left(lst, i)
                # we need the (idx + k - 1)-th element in lst to exist
                tgt = idx + k - 1
                if tgt < len(lst):
                    # this character reaches k occurrences at position lst[tgt]
                    min_j = min(min_j, lst[tgt])
            # If we found some character that gets k occurrences,
            # then any substring s[i..j] with j ≥ min_j is valid.
            if min_j < INF:
                ans += (n - min_j)
        
        return ans