from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Precompute the Longest Common Prefix (LCP) array
        lcp = [[0] * n for _ in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(n-1, -1, -1):
                if nums[i] == nums[j]:
                    if i + 1 < n and j + 1 < n:
                        lcp[i][j] = 1 + lcp[i+1][j+1]
                    else:
                        lcp[i][j] = 1
                else:
                    lcp[i][j] = 0
        
        count = 0
        for i in range(n-1):  # i can be from 0 to n-2
            for j in range(i+1, n-1):  # j can be from i+1 to n-2
                # Check case 1: nums1 is prefix of nums2
                case1 = False
                if j >= 2 * i + 1:
                    len1 = i + 1
                    # Check if the first i+1 elements of nums1 match the first i+1 elements of nums2
                    if lcp[0][i+1] >= len1:
                        case1 = True
                
                # Check case 2: nums2 is prefix of nums3
                case2 = False
                len2 = j - i
                len3 = (n - 1) - j
                if len2 <= len3:
                    # Check if nums2 is a prefix of nums3
                    if lcp[i+1][j+1] >= len2:
                        case2 = True
                
                if case1 or case2:
                    count += 1
        
        return count