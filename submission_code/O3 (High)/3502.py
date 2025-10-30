class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        """
        Count substrings that contain at least one character
        appearing at least k times.

        Idea:
            Total substrings = n*(n+1)//2
            We subtract the number of “bad” substrings in which every
            character appears at most k-1 times.

            The set of “bad” substrings is monotone with window size:
            if a window fulfils the restriction (all counts ≤ k-1),
            every sub-window also does.  
            Hence we can count them with a sliding window that keeps
            the restriction satisfied.

        Complexity:
            Time  : O(n)   (n ≤ 3000 by constraints)
            Memory: O(1)   (26 counters)
        """
        n = len(s)
        total = n * (n + 1) // 2                     # all substrings
        
        # k == 1 ⇒ no substring can be “bad”
        if k == 1:
            return total
        
        cnt = [0] * 26                               # letter counts
        left = 0
        bad = 0                                      # substrings with all counts ≤ k-1
        limit = k - 1                                # maximal allowed frequency
        
        for right, ch in enumerate(s):
            idx = ord(ch) - 97
            cnt[idx] += 1
            
            # shrink until constraint is satisfied again (cnt[idx] ≤ limit)
            while cnt[idx] > limit:
                cnt[ord(s[left]) - 97] -= 1
                left += 1
            
            # all substrings ending at 'right' and starting in [left .. right]
            # are “bad”; there are (right - left + 1) of them
            bad += right - left + 1
        
        # wanted substrings = total – bad ones
        return total - bad