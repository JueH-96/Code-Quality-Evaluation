class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # To store the minimum prefix sum for each possible starting value
        prefix_min = defaultdict(lambda: float('inf'))
        prefix_sum = 0
        max_sum = -float('inf')
        
        for num in nums:
            prefix_sum += num
            # Check for num - k
            if (num - k) in prefix_min:
                current_sum = prefix_sum - prefix_min[num - k]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Check for num + k
            if (num + k) in prefix_min:
                current_sum = prefix_sum - prefix_min[num + k]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Update the prefix_min for the current num
            if prefix_sum < prefix_min[num]:
                prefix_min[num] = prefix_sum
        
        return max_sum if max_sum != -float('inf') else 0