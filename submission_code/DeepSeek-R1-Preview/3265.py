class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        min_prefix = {}
        max_sum = -float('inf')
        
        for j in range(len(nums)):
            num = nums[j]
            next_prefix_sum = prefix_sum + num
            x1 = num + k
            x2 = num - k
            current_max = -float('inf')
            
            # Check for x1 and x2 in min_prefix
            if x1 in min_prefix:
                current_max = next_prefix_sum - min_prefix[x1]
            if x2 in min_prefix:
                candidate = next_prefix_sum - min_prefix[x2]
                if candidate > current_max:
                    current_max = candidate
            
            # Update max_sum if a valid candidate is found
            if current_max != -float('inf'):
                if current_max > max_sum:
                    max_sum = current_max
            
            # Update min_prefix for the current number
            if num in min_prefix:
                if prefix_sum < min_prefix[num]:
                    min_prefix[num] = prefix_sum
            else:
                min_prefix[num] = prefix_sum
            
            # Move to the next prefix sum
            prefix_sum = next_prefix_sum
        
        # Return 0 if no good subarray found, else return the max_sum
        return max_sum if max_sum != -float('inf') else 0