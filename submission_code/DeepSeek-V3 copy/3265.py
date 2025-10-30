class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # To store the minimum prefix sum for each possible starting element
        prefix_min = defaultdict(lambda: float('inf'))
        max_sum = -float('inf')
        current_prefix = 0
        
        for num in nums:
            current_prefix += num
            # Check for num + k and num - k
            if (num + k) in prefix_min:
                max_sum = max(max_sum, current_prefix - prefix_min[num + k])
            if (num - k) in prefix_min:
                max_sum = max(max_sum, current_prefix - prefix_min[num - k])
            # Update the prefix_min for the current num
            if current_prefix < prefix_min[num]:
                prefix_min[num] = current_prefix
        
        return max_sum if max_sum != -float('inf') else 0