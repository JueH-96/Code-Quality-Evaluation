class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = 0
        s2 = 0
        count1 = 0
        count2 = 0
        
        # Calculate sum of nonzero elements and count zeros
        for x in nums1:
            if x == 0:
                count1 += 1
            else:
                s1 += x
        for y in nums2:
            if y == 0:
                count2 += 1
            else:
                s2 += y
        
        # If neither array has zeros:
        if count1 == 0 and count2 == 0:
            # Sums must already match or it's impossible
            return s1 if s1 == s2 else -1
        
        # If nums1 has no zeros, all "replacement sum" must go to nums2
        if count1 == 0 and count2 > 0:
            # s1 == s2 + X2  =>  X2 = s1 - s2
            X2 = s1 - s2
            # Must have X2 >= count2 (filling count2 zeros with strictly positive integers)
            # And X2 must be nonnegative
            if X2 < count2 or X2 < 0:
                return -1
            return s1  # s1 + 0 == s2 + X2, final sums both = s1
        
        # If nums2 has no zeros, all "replacement sum" must go to nums1
        if count2 == 0 and count1 > 0:
            # s1 + X1 == s2  =>  X1 = s2 - s1
            X1 = s2 - s1
            # Must have X1 >= count1 and X1 >= 0
            if X1 < count1 or X1 < 0:
                return -1
            return s2  # final sums both = s2
        
        # Now, both arrays have at least one zero
        delta = s1 - s2
        # Case 1: sums are equal so far
        if delta == 0:
            # We just need to make replacements so that both sides remain equal
            # Each side must add at least count1 or count2, but the additions must match
            # Minimum is max(count1, count2)
            return s1 + max(count1, count2)
        
        # Case 2: s1 > s2
        if delta > 0:
            # We want X2 - X1 = delta
            # Let X1 = x, X2 = x + delta
            # x >= count1, and x + delta >= count2
            x = max(count1, count2 - delta)
            # x must be >= 0 anyway, but max(count1, ...) ensures x>=count1>=0 if needed
            if x < 0:
                return -1  # Should never happen in practice with count1>=0, but just in case
            # Check X2 >= count2
            if x + delta < count2:
                return -1
            return s1 + x
        
        # Case 3: s2 > s1 (delta < 0)
        # Let d = -delta = (s2 - s1) > 0
        d = -delta
        # We want X1 - X2 = delta => X1 - X2 = -d => X2 - X1 = d
        # Alternatively, s2 + X2 = s1 + X1 => s2 - s1 = X1 - X2 => d = X1 - X2 => X1 = X2 + d
        # Let X2 = x, then X1 = x + d
        # x >= count2, x + d >= count1
        x = max(count2, count1 - d)
        if x < 0:
            return -1
        if x + d < count1:
            return -1
        return s2 + x