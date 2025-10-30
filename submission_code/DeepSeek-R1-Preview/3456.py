from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0
        
        dp = {}
        first_num = nums[0]
        dp[(0, first_num)] = 1
        
        for num in nums[1:]:
            temp_dp = {}
            
            # Process existing states
            for (c, last), length in dp.items():
                if num == last:
                    new_c = c
                    new_length = length + 1
                    key = (new_c, num)
                    if key in temp_dp:
                        if new_length > temp_dp[key]:
                            temp_dp[key] = new_length
                    else:
                        temp_dp[key] = new_length
                else:
                    new_c = c + 1
                    if new_c > k:
                        continue
                    new_length = length + 1
                    key = (new_c, num)
                    if key in temp_dp:
                        if new_length > temp_dp[key]:
                            temp_dp[key] = new_length
                    else:
                        temp_dp[key] = new_length
            
            # Add the new subsequence starting with current num
            key = (0, num)
            if key in temp_dp:
                if 1 > temp_dp[key]:
                    temp_dp[key] = 1
            else:
                temp_dp[key] = 1
            
            # Merge temp_dp into the new_dp
            new_dp = dp.copy()
            for key_td, val_td in temp_dp.items():
                if key_td in new_dp:
                    if val_td > new_dp[key_td]:
                        new_dp[key_td] = val_td
                else:
                    new_dp[key_td] = val_td
            
            dp = new_dp
        
        # Find the maximum length where c <= k
        max_length = 0
        for (c, _), length in dp.items():
            if c <= k and length > max_length:
                max_length = length
        
        return max_length