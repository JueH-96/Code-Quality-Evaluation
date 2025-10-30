from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        # Step 1: Pre-compute LCP (Longest Common Prefix) table
        # lcp[i][j] will store the length of the longest common prefix
        # between the suffix starting at index i and the suffix starting at index j.
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if nums[i] == nums[j]:
                    lcp[i][j] = 1 + lcp[i + 1][j + 1]

        # The problem asks for the number of ways to split nums into three non-empty
        # subarrays nums1, nums2, nums3. This corresponds to choosing two split points.
        # Let the first split be after index `i` and the second after index `j`.
        # nums1 = nums[0...i], nums2 = nums[i+1...j], nums3 = nums[j+1...n-1]
        # The split points (i, j) must satisfy 0 <= i < j < n-1.
        
        # A split is "beautiful" if:
        # Cond1: nums1 is a prefix of nums2, OR
        # Cond2: nums2 is a prefix of nums3.

        # We use the Principle of Inclusion-Exclusion:
        # Total = (count for Cond1) + (count for Cond2) - (count for both)
        # Let C1 = count for Cond1, C2 = count for Cond2, C12 = count for both.

        # Cond1: nums1 is a prefix of nums2
        # This requires two things:
        # 1. len(nums1) <= len(nums2) => i + 1 <= j - i => j >= 2*i + 1
        # 2. The prefix matches => lcp[0][i+1] >= len(nums1) => lcp[0][i+1] >= i + 1
        
        # Cond2: nums2 is a prefix of nums3
        # This requires:
        # 1. len(nums2) <= len(nums3) is implicitly checked by the lcp condition.
        # 2. The prefix matches => lcp[i+1][j+1] >= len(nums2) => lcp[i+1][j+1] >= j - i

        c1 = 0
        c2 = 0
        c12 = 0

        # i loops from 0 to n-3
        for i in range(n - 2):
            len1 = i + 1
            
            # Check the part of Cond1 that only depends on i
            cond1_prefix_ok = lcp[0][i + 1] >= len1
            j_min_for_cond1 = 2 * i + 1

            # Calculate C1 contribution for this i.
            # If cond1_prefix_ok, we count all j's from j_min_for_cond1 to n-2.
            if cond1_prefix_ok and j_min_for_cond1 <= n - 2:
                c1 += (n - 2) - j_min_for_cond1 + 1

            # j loops from i+1 to n-2
            for j in range(i + 1, n - 1):
                len2 = j - i
                
                # Check Cond2
                cond2_ok = lcp[i + 1][j + 1] >= len2
                
                if cond2_ok:
                    c2 += 1
                    # If Cond2 is met, check if Cond1 is also met for C12
                    if cond1_prefix_ok and j >= j_min_for_cond1:
                        c12 += 1
        
        return c1 + c2 - c12