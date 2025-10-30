from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        max_attach = [[0 for _ in range(300)] for _ in range(301)]  # num 1-300, d 0-299
        ans = 1
        
        for i in range(n):
            dp_i = [0] * 300  # dp for current i, d 0-299
            num_i = nums[i]
            
            # Compute dp_i[d] for each d
            for d in range(300):
                num_j1 = num_i - d
                num_j2 = num_i + d
                max_val = -1  # invalid
                if 1 <= num_j1 <= 300 and max_attach[num_j1][d] > 0:
                    max_val = max(max_val, max_attach[num_j1][d])
                if 1 <= num_j2 <= 300 and max_attach[num_j2][d] > 0:
                    max_val = max(max_val, max_attach[num_j2][d])
                if max_val != -1:
                    dp_i[d] = 1 + max_val
            
            # Compute max ending at i
            max_end_i = 1
            for length in dp_i:
                if length > 0:
                    max_end_i = max(max_end_i, length)
            ans = max(ans, max_end_i)
            
            # Compute suffix max for dp_i
            suffix_max_array = [0] * 300
            max_so_far = 0
            for k in range(299, -1, -1):  # from 299 to 0
                if dp_i[k] > 0:
                    max_so_far = max(max_so_far, dp_i[k])
                suffix_max_array[k] = max_so_far  # max over d' >= k of dp_i[d']
            
            # Compute attachable length
            attach_len = [max(1, suff) for suff in suffix_max_array]
            
            # Update max_attach for num_i
            for d in range(300):
                max_attach[num_i][d] = max(max_attach[num_i][d], attach_len[d])
        
        return ans