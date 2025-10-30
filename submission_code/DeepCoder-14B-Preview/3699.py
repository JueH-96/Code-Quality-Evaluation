import bisect
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: list) -> int:
        n = len(nums)
        if n < 4:
            return 0
        
        # Preprocess the array to map each value to its list of indices
        value_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            value_indices[num].append(idx)
        
        total = 0
        
        for r in range(2, n - 1):
            for s in range(r + 2, n):
                # Iterate over all possible q's that are at least two positions before r
                for q in range(2, r - 1 + 1):  # r-1 is exclusive, so +1 to include r-2
                    numerator = nums[q] * nums[s]
                    denominator = nums[r]
                    if denominator == 0:
                        continue  # Since nums contains positive integers, this can't happen
                    if numerator % denominator != 0:
                        continue
                    required_p_value = numerator // denominator
                    
                    # Get the list of indices for the required_p_value
                    if required_p_value in value_indices:
                        lst = value_indices[required_p_value]
                        # Find the number of elements less than q-1
                        count_p = bisect.bisect_left(lst, q - 1)
                        total += count_p
        
        return total