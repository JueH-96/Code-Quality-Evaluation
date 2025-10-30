from typing import List
from collections import defaultdict
import math

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        # Step 1: Count the frequency of each element
        frequency = defaultdict(int)
        for num in nums:
            frequency[num] += 1

        # Step 2: Find the maximum frequency
        max_frequency = max(frequency.values())

        # Step 3: Calculate the minimum number of operations
        # The minimum operations needed is the ceiling of half the size of the array minus the maximum frequency
        n = len(nums)
        min_operations = math.ceil(n / 2) - max_frequency

        return min_operations