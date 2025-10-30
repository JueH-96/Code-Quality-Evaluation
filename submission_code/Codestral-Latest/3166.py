class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        from math import gcd
        from functools import reduce

        # Count the frequency of each number in nums
        freq = Counter(nums)

        # Find the minimum frequency
        min_freq = min(freq.values())

        # Helper function to check if a given number of groups is valid
        def is_valid_groups(groups):
            for count in freq.values():
                if count % groups != count // groups:
                    return False
            return True

        # Find the greatest common divisor of all frequencies
        gcd_freq = reduce(gcd, freq.values())

        # Check from gcd_freq down to 1 to find the minimum valid number of groups
        for groups in range(gcd_freq, 0, -1):
            if is_valid_groups(groups):
                return sum((count + groups - 1) // groups for count in freq.values())

        return len(nums)