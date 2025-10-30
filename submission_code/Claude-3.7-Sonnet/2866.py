class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        
        for l in range(n):
            # Check if the starting element is even and <= threshold
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_length = 1
                
                # Expand the subarray as far as possible
                for r in range(l+1, n):
                    # Check if current element is <= threshold and has different parity from previous
                    if nums[r] <= threshold and nums[r] % 2 != nums[r-1] % 2:
                        current_length += 1
                    else:
                        break
                
                # Update maximum length found
                max_length = max(max_length, current_length)
        
        return max_length