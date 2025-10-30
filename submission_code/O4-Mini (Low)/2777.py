from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        # suffix_count[i] = number of distinct elements in nums[i:]
        suffix_count = [0] * (n + 1)
        seen_suffix = set()
        
        # Build suffix distinct counts from the end
        for i in range(n - 1, -1, -1):
            seen_suffix.add(nums[i])
            suffix_count[i] = len(seen_suffix)
        
        result = []
        seen_prefix = set()
        prefix_distinct = 0
        
        # For each index i, prefix is nums[0..i], suffix is nums[i+1..]
        for i in range(n):
            # add current to prefix
            if nums[i] not in seen_prefix:
                seen_prefix.add(nums[i])
                prefix_distinct += 1
            
            # suffix distinct count for nums[i+1..] is suffix_count[i+1]
            suf = suffix_count[i + 1]
            result.append(prefix_distinct - suf)
        
        return result