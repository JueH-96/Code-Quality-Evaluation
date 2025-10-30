class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        n = len(nums)
        positions = defaultdict(list)
        
        # Group positions by values
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        # Calculate prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_sum = float('-inf')
        found_good_subarray = False
        
        for value in positions:
            for target in [value + k, value - k]:
                if target in positions:
                    for start_pos in positions[value]:
                        for end_pos in positions[target]:
                            if end_pos > start_pos:
                                current_sum = prefix[end_pos + 1] - prefix[start_pos]
                                max_sum = max(max_sum, current_sum)
                                found_good_subarray = True
        
        return max_sum if found_good_subarray else 0