class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Precompute the prefix lengths
        prefix_len = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                l = 0
                while a + l < n and b + l < n and nums[a + l] == nums[b + l]:
                    l += 1
                prefix_len[a][b] = l
        
        count = 0
        for i in range(1, n - 1):  # i can be from 1 to n-2 inclusive
            for j in range(i + 1, n):  # j can be from i+1 to n-1 inclusive
                # Condition 1: nums1 is a prefix of nums2
                cond1 = False
                if j >= 2 * i:
                    if prefix_len[0][i] >= i:
                        cond1 = True
                
                # Condition 2: nums2 is a prefix of nums3
                cond2 = False
                if (n - j) >= (j - i):
                    if prefix_len[i][j] >= (j - i):
                        cond2 = True
                
                if cond1 or cond2:
                    count += 1
        
        return count