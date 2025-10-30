class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        # Precompute the bitwise OR of all elements
        total_or = 0
        for num in nums:
            total_or |= num
        
        max_or = total_or
        
        # Iterate through each element to consider applying the operation
        for i in range(len(nums)):
            # Apply the operation k times on nums[i]
            current = nums[i] << k
            # Compute the new OR value
            new_or = total_or ^ (nums[i] & total_or)  # Remove the original nums[i]'s contribution
            new_or |= current  # Add the new value's contribution
            if new_or > max_or:
                max_or = new_or
        
        return max_or