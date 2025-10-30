from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # create prefix sum array, where prefix[i+1] is sum of nums[0..i]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # dictionary to store minimum prefix sum corresponding to a candidate start element value.
        # For a start index i, the candidate key is nums[i] and value is min(prefix[i]) among those indices.
        best_prefix = {}
        max_sum = None  # We'll track if we found any good subarray
        
        for j in range(n):
            # as a last element, j must be at least 1 (subarray length at least 2), so dictionary contains indices < j.
            if j > 0:
                # For a subarray to be good: |nums[i] - nums[j]| == k.
                # That means nums[i] == nums[j] + k or nums[i] == nums[j] - k.
                for candidate in (nums[j] + k, nums[j] - k):
                    if candidate in best_prefix:
                        # subarray sum = prefix[j+1] - prefix[i]
                        cur_sum = prefix[j + 1] - best_prefix[candidate]
                        if max_sum is None or cur_sum > max_sum:
                            max_sum = cur_sum
            # Now update dictionary with nums[j] as a potential start element.
            # The prefix sum value corresponding to starting element at index j is prefix[j].
            # We want the smallest prefix[j] (to maximize the future subarray sum) for this value.
            if nums[j] in best_prefix:
                if prefix[j] < best_prefix[nums[j]]:
                    best_prefix[nums[j]] = prefix[j]
            else:
                best_prefix[nums[j]] = prefix[j]
                
        return max_sum if max_sum is not None else 0