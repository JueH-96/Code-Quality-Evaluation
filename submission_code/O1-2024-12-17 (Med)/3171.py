class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        import math
        
        # Calculate the sum of non-zero elements and count the zeros for each array
        s1, n1 = 0, 0
        for x in nums1:
            if x == 0:
                n1 += 1
            else:
                s1 += x
                
        s2, n2 = 0, 0
        for x in nums2:
            if x == 0:
                n2 += 1
            else:
                s2 += x
        
        # If neither array has zeros, check if sums already match
        if n1 == 0 and n2 == 0:
            return s1 if s1 == s2 else -1
        
        # If nums1 has no zeros, we cannot change its sum. We must adjust nums2 to match s1.
        # sum_in_nums2 = s2 + something = s1  => something = s1 - s2
        if n1 == 0:
            # let d = s2 - s1 => we need sum_of_y = -d = s1 - s2
            d = s2 - s1
            needed = -d  # s1 - s2
            # needed must be at least n2 (since we need to distribute among n2 zeros, each >=1)
            # and non-negative
            if needed >= n2 and needed >= 0:
                return s1   # sums become equal at s1
            else:
                return -1
        
        # If nums2 has no zeros, we try to match s2 by filling zeros in nums1
        # sum_in_nums1 = s1 + sum_of_x = s2 => sum_of_x = s2 - s1
        if n2 == 0:
            d = s2 - s1
            if d >= n1:
                return s2
            else:
                return -1
        
        # Otherwise, both arrays have zeros. Let d = s2 - s1.
        # We want s1 + X = s2 + Y with X - Y = d, X >= n1, Y >= n2.
        # Set X = max(n1, n2 + d). Then Y = X - d, and the sum is s1 + X.
        
        d = s2 - s1
        X = max(n1, n2 + d)  # minimal sum of zero-replacements in nums1
        if X < 0:
            return -1
        Y = X - d            # sum of zero-replacements in nums2
        if Y < 0:
            return -1
        
        # At this point, X >= n1 and Y >= n2 are guaranteed by construction if feasible.
        return s1 + X