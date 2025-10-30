class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix OR array
        prefix_or = [0] * n
        prefix_or[0] = nums[0]
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i]
        
        # Precompute suffix OR array
        suffix_or = [0] * n
        suffix_or[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i]
        
        max_or = 0
        
        # Try multiplying each element by 2^k
        for i in range(n):
            # Calculate OR of all elements except the current one
            or_without_i = (prefix_or[i-1] if i > 0 else 0) | (suffix_or[i+1] if i < n-1 else 0)
            
            # Calculate new OR with the current element multiplied by 2^k
            new_or = or_without_i | (nums[i] << k)
            
            max_or = max(max_or, new_or)
        
        return max_or