class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        # Precompute the prefix and suffix ORs
        n = len(nums)
        prefix = [0] * n
        suffix = [0] * n
        
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] | nums[i]
        
        suffix[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suffix[i] = suffix[i+1] | nums[i]
        
        max_or = 0
        for i in range(n):
            # Calculate the OR without nums[i]
            current_or = 0
            if i > 0:
                current_or |= prefix[i-1]
            if i < n-1:
                current_or |= suffix[i+1]
            # Apply the operation k times on nums[i]
            num = nums[i] << k
            # Combine with the rest
            total_or = current_or | num
            if total_or > max_or:
                max_or = total_or
        return max_or