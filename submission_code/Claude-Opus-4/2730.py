class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix and suffix ORs for efficiency
        prefix = [0] * n
        suffix = [0] * n
        
        # Calculate prefix ORs
        for i in range(n):
            if i == 0:
                prefix[i] = 0
            else:
                prefix[i] = prefix[i-1] | nums[i-1]
        
        # Calculate suffix ORs
        for i in range(n-1, -1, -1):
            if i == n-1:
                suffix[i] = 0
            else:
                suffix[i] = suffix[i+1] | nums[i+1]
        
        max_or = 0
        
        # Try applying all k operations to each element
        for i in range(n):
            # Apply k operations to nums[i] (multiply by 2^k)
            modified_num = nums[i] << k
            
            # Calculate the OR with this modification
            current_or = prefix[i] | modified_num | suffix[i]
            
            max_or = max(max_or, current_or)
        
        return max_or