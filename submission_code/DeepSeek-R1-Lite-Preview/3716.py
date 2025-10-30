from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # Initialize global_max for differences 0 to 300
        global_max = [0] * 301
        answer = 1

        # To keep track of frequency of each value up to current index
        from collections import defaultdict
        freq = defaultdict(int)

        for i in range(n):
            # Compute suffix_max from global_max
            suffix_max = [0] * 301
            suffix_max[300] = global_max[300]
            for d in range(299, -1, -1):
                suffix_max[d] = max(global_max[d], suffix_max[d+1])
            
            # Iterate through all possible previous values v
            for v in freq:
                diff = abs(nums[i] - v)
                dp_i_diff = suffix_max[diff] + 1
                if dp_i_diff > global_max[diff]:
                    global_max[diff] = dp_i_diff
                    answer = max(answer, dp_i_diff)
            
            # Update frequency of current value
            freq[nums[i]] += 1

        return answer