class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        from collections import Counter
        from math import gcd
        from functools import reduce

        # Helper function to find the GCD of a list of numbers
        def find_gcd(lst):
            return reduce(gcd, lst)

        # Count the frequency of each number in the array
        freq = Counter(nums)

        # Find the GCD of all the numbers in the array
        gcd_all = find_gcd(nums)

        # If the GCD is 1, we can reduce the array to a single element
        if gcd_all == 1:
            return 1

        # Otherwise, the minimum length is the number of distinct numbers
        # that are multiples of the GCD
        return len(freq)