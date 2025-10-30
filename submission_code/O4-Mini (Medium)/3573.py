class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        # Edge case: if word2 is longer than word1, no substrings can satisfy
        if m > n:
            return 0
        
        # Build the requirement counts for word2
        need = [0] * 26
        for c in word2:
            need[ord(c) - ord('a')] += 1
        # missing = total characters still needed in the current window
        missing = sum(need)
        
        # Sliding window counts for the current window word1[l:r]
        window = [0] * 26
        ans = 0
        r = 0
        
        for l in range(n):
            # Expand r until we either satisfy all needs or run out of string
            while r < n and missing > 0:
                idx = ord(word1[r]) - ord('a')
                # If this character is still needed, decrement missing
                if window[idx] < need[idx]:
                    missing -= 1
                window[idx] += 1
                r += 1
            
            # If we've satisfied all needs, every extension up to the end is valid
            if missing == 0:
                # r-1 is the minimal end index that satisfies needs
                # Any end in [r-1 .. n-1] works, so we add (n - (r-1)) substrings
                ans += n - r + 1
            else:
                # We can't satisfy for this l or any larger l
                break
            
            # Before moving l -> l+1, remove word1[l] from the window
            idx_l = ord(word1[l]) - ord('a')
            window[idx_l] -= 1
            # If after removal we're below the requirement, we become missing one more
            if window[idx_l] < need[idx_l]:
                missing += 1
        
        return ans