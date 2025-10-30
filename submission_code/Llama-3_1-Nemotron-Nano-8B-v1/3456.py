from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0
        
        # Initialize DP with the first element
        dp = {(nums[0], 0): 1}
        
        for i in range(1, len(nums)):
            curr = nums[i]
            temp = dict(dp)  # Start with previous states carried forward
            
            # Process each previous state to see if we can append the current element
            for (prev_val, prev_t) in list(dp.keys()):
                if prev_val == curr:
                    new_t = prev_t
                    new_len = dp[(prev_val, prev_t)] + 1
                    if (curr, new_t) in temp:
                        if new_len > temp[(curr, new_t)]:
                            temp[(curr, new_t)] = new_len
                    else:
                        temp[(curr, new_t)] = new_len
                else:
                    new_t = prev_t + 1
                    if new_t > k:
                        continue
                    new_len = dp[(prev_val, prev_t)] + 1
                    if (curr, new_t) in temp:
                        if new_len > temp[(curr, new_t)]:
                            temp[(curr, new_t)] = new_len
                    else:
                        temp[(curr, new_t)] = new_len
            
            # Consider starting a new subsequence with the current element
            if (curr, 0) not in temp or 1 > temp.get((curr, 0), 0):
                temp[(curr, 0)] = 1
            
            # Update DP to the new state
            dp = temp
        
        # The maximum length is the maximum value in the DP dictionary
        return max(dp.values()) if dp else 0