from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Precompute the LCP array
        lcp = [[0] * n for _ in range(n)]
        for a in range(n-1, -1, -1):
            for b in range(n-1, -1, -1):
                if nums[a] == nums[b]:
                    if a + 1 < n and b + 1 < n:
                        lcp[a][b] = 1 + lcp[a+1][b+1]
                    else:
                        lcp[a][b] = 1
                else:
                    lcp[a][b] = 0
        
        count = 0
        for i in range(n-2):  # i can be from 0 to n-3
            for j in range(i+1, n-1):  # j can be from i+1 to n-2
                # Check condition 1: nums1 is a prefix of nums2
                condition1 = False
                if j >= 2 * i + 1:
                    if lcp[0][i+1] >= (i + 1):
                        condition1 = True
                
                # Check condition 2: nums2 is a prefix of nums3
                condition2 = False
                if 2 * j <= (n - 1 + i):
                    if lcp[i+1][j+1] >= (j - i):
                        condition2 = True
                
                if condition1 or condition2:
                    count += 1
        
        return count