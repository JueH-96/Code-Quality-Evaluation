class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        
        for i in range(n):
            # Condition 1: subarray starts with an even number
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                length = 1
                last_parity = nums[i] % 2
                
                # Extend subarray while conditions hold
                for j in range(i+1, n):
                    if nums[j] <= threshold and (nums[j] % 2 != last_parity):
                        length += 1
                        last_parity = nums[j] % 2
                    else:
                        break
                
                max_len = max(max_len, length)
        
        return max_len