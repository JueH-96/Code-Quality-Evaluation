class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix OR array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
            
        # Calculate suffix OR array
        suffix = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
            
        # Try multiplying each number by 2^k and find max OR
        result = 0
        for i in range(n):
            # Calculate OR value when nums[i] is multiplied by 2^k
            curr = prefix[i] | (nums[i] << k) | suffix[i + 1]
            result = max(result, curr)
            
        return result