class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        min_prefix = {}
        max_sum = -float('inf')
        
        for num in nums:
            current_prefix = prefix_sum + num
            # Check possible values (num + k and num - k)
            for val in [num + k, num - k]:
                if val in min_prefix:
                    candidate = current_prefix - min_prefix[val]
                    if candidate > max_sum:
                        max_sum = candidate
            # Update min_prefix for the current num with the prefix_sum before adding this num
            if num not in min_prefix:
                min_prefix[num] = prefix_sum
            else:
                if prefix_sum < min_prefix[num]:
                    min_prefix[num] = prefix_sum
            # Update prefix_sum for the next iteration
            prefix_sum = current_prefix
        
        return max_sum if max_sum != -float('inf') else 0