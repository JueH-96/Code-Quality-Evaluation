class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        
        # Compute prefix OR array
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
        
        # Compute suffix OR array
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        
        max_or = 0
        for i in range(n):
            # Calculate the OR when applying all k operations to nums[i]
            shifted = nums[i] << k
            # OR of all elements except nums[i]
            or_without_i = prefix[i] | suffix[i + 1]
            current_or = shifted | or_without_i
            if current_or > max_or:
                max_or = current_or
        
        return max_or