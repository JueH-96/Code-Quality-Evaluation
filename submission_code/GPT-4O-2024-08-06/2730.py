class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate the initial OR of the entire array
        current_or = 0
        for num in nums:
            current_or |= num
        
        # Try to maximize the OR by doubling each element up to k times
        max_or = current_or
        for i in range(n):
            # Calculate the OR if we double nums[i] up to k times
            doubled_value = nums[i] * (2 ** k)
            new_or = current_or | doubled_value
            
            # Update the maximum OR found
            max_or = max(max_or, new_or)
        
        return max_or