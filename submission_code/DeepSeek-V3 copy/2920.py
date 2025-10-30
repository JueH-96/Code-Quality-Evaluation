from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if all(x == nums[0] for x in nums):
            return 0
        # Create a dictionary to map each value to its indices
        value_to_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            value_to_indices[num].append(idx)
        min_seconds = float('inf')
        for num in value_to_indices:
            indices = value_to_indices[num]
            max_gap = 0
            # Calculate the maximum gap between consecutive indices
            for i in range(len(indices)):
                current_gap = (indices[(i+1)%len(indices)] - indices[i] - 1) % n
                if current_gap < 0:
                    current_gap += n
                max_gap = max(max_gap, current_gap)
            # The number of seconds required is the ceiling of (max_gap / 2)
            seconds = (max_gap + 1) // 2
            min_seconds = min(min_seconds, seconds)
        return min_seconds