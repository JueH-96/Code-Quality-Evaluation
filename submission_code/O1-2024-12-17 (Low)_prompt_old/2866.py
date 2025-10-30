class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        
        # We try to start a valid subarray from each index
        for start in range(n):
            # Condition 1: nums[start] must be even and <= threshold
            if nums[start] % 2 == 0 and nums[start] <= threshold:
                length = 1
                prev_parity = nums[start] % 2
                
                # Extend the subarray as far as possible
                for j in range(start + 1, n):
                    # Condition 3: each element must be <= threshold
                    # Condition 2: adjacent parities must alternate
                    if nums[j] <= threshold and (nums[j] % 2 != prev_parity):
                        length += 1
                        prev_parity = nums[j] % 2
                    else:
                        break
                
                max_len = max(max_len, length)
        
        return max_len