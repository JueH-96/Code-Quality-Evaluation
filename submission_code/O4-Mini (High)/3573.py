class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # word1: the main string, word2: the prefix we need to be able to form
        n = len(word1)
        
        # Count requirements from word2
        req = [0] * 26
        for c in word2:
            req[ord(c) - 97] += 1
        
        # How many distinct letters we actually need to satisfy
        need = sum(1 for x in req if x > 0)
        
        # Sliding window over word1
        curr = [0] * 26       # current window counts
        satisfied = 0         # how many of the needed letters meet their req
        ans = 0
        l = 0
        r = 0
        
        while l < n:
            # Expand r until we satisfy all required counts or run out of string
            while r < n and satisfied < need:
                idx = ord(word1[r]) - 97
                curr[idx] += 1
                # If this increment makes curr[idx] hit req[idx], we mark that requirement satisfied
                if req[idx] > 0 and curr[idx] == req[idx]:
                    satisfied += 1
                r += 1
            
            # If we still haven't satisfied all requirements, no further windows will work
            if satisfied < need:
                break
            
            # Now [l..r-1] is the minimal window starting at l that works.
            # Any extension to the right will also work: that's (n - (r-1)) = n - r + 1 substrings.
            ans += (n - r + 1)
            
            # Move left pointer up by one, adjusting counts & satisfaction
            idx_l = ord(word1[l]) - 97
            # If curr[idx_l] was exactly meeting req[idx_l], removing it will break that satisfaction
            if req[idx_l] > 0 and curr[idx_l] == req[idx_l]:
                satisfied -= 1
            curr[idx_l] -= 1
            l += 1
        
        return ans