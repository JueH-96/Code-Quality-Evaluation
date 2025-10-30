class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        
        # Initialize the minimum number of operations to the maximum possible value
        min_ops = float('inf')
        
        # Iterate through all possible subarrays
        for i in range(n):
            for j in range(i, n):
                # Compute the difference between the current subarray and the target subarray
                diff = sum(nums[i:j+1]) - sum(target[i:j+1])
                
                # If the difference is 0, we have found the minimum number of operations
                if diff == 0:
                    min_ops = min(min_ops, j - i + 1)
                
                # If the difference is positive, we can decrement the subarray by 1 to make it equal to the target
                elif diff > 0:
                    min_ops = min(min_ops, j - i + 1 + diff)
                
                # If the difference is negative, we can increment the subarray by 1 to make it equal to the target
                else:
                    min_ops = min(min_ops, j - i + 1 - diff)
        
        # If no subarray can be made equal to the target, return -1
        return min_ops if min_ops != float('inf') else -1