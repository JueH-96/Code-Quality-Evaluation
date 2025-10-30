from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # Convert numbers to strings and pad with leading zeros
        max_len = len(str(max(nums)))
        padded_nums = [str(num).zfill(max_len) for num in nums]
        
        def count_differences(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        count = 0
        n = len(padded_nums)
        for i in range(n):
            for j in range(i + 1, n):
                if sorted(padded_nums[i]) == sorted(padded_nums[j]):
                    diff = count_differences(padded_nums[i], padded_nums[j])
                    if diff <= 2:
                        count += 1
        return count