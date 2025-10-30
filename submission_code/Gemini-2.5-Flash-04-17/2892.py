from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Calculate the expected value of n if nums is a permutation of base[n].
        # The length of base[n] is n + 1. So, n must be len(nums) - 1.
        n = len(nums) - 1

        # Base case: A good array must be a permutation of base[n] for n >= 1.
        # The smallest base[n] is base[1] = [1, 1], which has length 2.
        # If nums has length less than 2, it cannot be a good array.
        if n < 1: # This is equivalent to len(nums) < 2
            return False

        # We need to check the frequencies of numbers in nums.
        # A good array (permutation of base[n]) must contain numbers 1 to n-1 exactly once,
        # and the number n exactly twice.
        # We can use a frequency counter. Since the expected numbers are 1 to n,
        # and n = len(nums) - 1 (with len(nums) <= 100), the maximum value of n is 99.
        # We can use a list of size n + 1 to store counts for numbers 0 to n.
        # Constraints guarantee num[i] >= 1, so we only care about indices 1 to n.
        # Using size n+1 is sufficient because we check if num > n during iteration.
        counts = [0] * (n + 1) # Initialize counts for numbers 0 up to n

        # Iterate through the input array nums to count frequencies.
        for num in nums:
            # If any number in nums is greater than the expected maximum value n,
            # it cannot be a permutation of base[n]. Return False immediately.
            if num > n:
                return False
            # Increment the count for the current number.
            # We don't need to check num >= 1 because constraints guarantee it.
            # Since num >= 1 and we checked num <= n, counts[num] is a valid index in [1, n].
            counts[num] += 1

        # Now, check if the frequencies match the definition of base[n]:
        # Numbers from 1 to n-1 must appear exactly once.
        # The range function `range(1, n)` correctly handles cases:
        # - If n=1, range(1, 1) is empty, loop doesn't run (correct as no numbers 1 to n-1).
        # - If n=2, range(1, 2) is 1, checks count[1] == 1.
        # - If n > 2, range(1, n) is 1, 2, ..., n-1, checks counts for each.
        for i in range(1, n):
            if counts[i] != 1:
                return False

        # The number n must appear exactly twice.
        # This check is only performed if the loop for 1 to n-1 completed without returning False.
        if counts[n] != 2:
            return False

        # If all the above checks pass, it means:
        # 1. All numbers in nums were within the range [1, n] (due to the num > n check).
        # 2. Numbers 1 through n-1 appeared exactly once (due to the loop check).
        # 3. Number n appeared exactly twice (due to the counts[n] check).
        # Since the total count of numbers 1 through n is (n-1)*1 + 2 = n+1,
        # and the length of nums is also n+1, and we only counted numbers <= n from nums,
        # the set of numbers in nums must be exactly {1 (once), ..., n-1 (once), n (twice)}.
        # Thus, nums is a permutation of base[n].
        return True