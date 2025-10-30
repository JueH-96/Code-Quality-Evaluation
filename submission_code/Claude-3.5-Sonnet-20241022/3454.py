class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        
        # Convert differences to prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + diff[i]
            
        # If final prefix sum is not 0, impossible to make equal
        if prefix[-1] != 0:
            return -1
            
        # Find minimum operations needed
        operations = 0
        min_prefix = float('inf')
        max_prefix = float('-inf')
        
        for p in prefix:
            min_prefix = min(min_prefix, p)
            max_prefix = max(max_prefix, p)
            
        return int(max_prefix - min_prefix)