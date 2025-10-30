class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        median_idx = (n - 1) // 2
        
        # Sort array to find elements that need to be changed
        sorted_nums = sorted(nums)
        
        # For odd length, we need middle element to be k
        # For even length, we need larger of two middle elements to be k
        if n % 2 == 0:
            median_idx = n // 2
            
        operations = 0
        
        # For elements before median index
        # If element > k, we need to decrease it to k
        # If element < k, we need to increase it to at most k
        for i in range(median_idx + 1):
            if sorted_nums[i] > k:
                operations += sorted_nums[i] - k
                
        # For elements after median index
        # If element < k, we need to increase it to k
        # If element > k, we can leave it as is
        for i in range(median_idx, n):
            if sorted_nums[i] < k:
                operations += k - sorted_nums[i]
                
        return operations