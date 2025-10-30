class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix OR (OR of all elements before index i)
        prefix_or = [0] * n
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i-1]
        
        # Calculate suffix OR (OR of all elements after index i)
        suffix_or = [0] * n
        for i in range(n-2, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i+1]
        
        max_or = 0
        # Try applying all k operations to each element
        for i in range(n):
            # OR of: (elements before i) | (nums[i] shifted by k) | (elements after i)
            result = prefix_or[i] | (nums[i] << k) | suffix_or[i]
            max_or = max(max_or, result)
        
        return max_or