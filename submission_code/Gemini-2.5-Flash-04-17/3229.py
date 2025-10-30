import bisect
from typing import List

class Solution:
    @staticmethod
    def generate_palindromes(limit: int) -> List[int]:
        """Generates a sorted list of positive palindromic integers less than limit."""
        palindromes = []

        # Iterate through the potential first halves of the palindrome.
        # Start from 1.
        first_half = 1
        while True:
            s_first_half = str(first_half)

            # Case 1: Odd length palindrome
            # Constructed by s_first_half + reverse(s_first_half[:-1])
            # For first_half = 1, s_first_half is '1', s_first_half[:-1] is '', reverse('') is '', palindrome is '1'.
            # For first_half = 10, s_first_half is '10', s_first_half[:-1] is '1', reverse('1') is '1', palindrome is '101'.
            # For first_half = 100, s_first_half is '100', s_first_half[:-1] is '10', reverse('10') is '01', palindrome is '10001'.
            s_rev_suffix_odd = s_first_half[:-1][::-1] if len(s_first_half) > 1 else ''
            palindrome_str_odd = s_first_half + s_rev_suffix_odd
            palindrome_odd = int(palindrome_str_odd)

            if palindrome_odd < limit:
                palindromes.append(palindrome_odd)


            # Case 2: Even length palindrome
            # Constructed by s_first_half + reverse(s_first_half)
            # For first_half = 1, s_first_half is '1', reverse('1') is '1', palindrome is '11'.
            # For first_half = 10, s_first_half is '10', reverse('10') is '01', palindrome is '1001'.
            # For first_half = 100, s_first_half is '100', reverse('100') is '001', palindrome is '100001'.
            s_rev_suffix_even = s_first_half[::-1]
            palindrome_str_even = s_first_half + s_rev_suffix_even
            palindrome_even = int(palindrome_str_even)

            # If the even palindrome is >= limit, then all subsequent palindromes
            # generated from larger `first_half` values will also be >= limit.
            # We can stop the loop.
            if palindrome_even >= limit:
                break # Stop the loop over first_half

            if palindrome_even < limit:
                palindromes.append(palindrome_even)

            # Move to the next first_half
            first_half += 1

        # Sorting is required as palindromes are not generated in strictly increasing order.
        # Example: 9 (from first_half=9, odd) < 11 (from first_half=1, even) < 101 (from first_half=10, odd).
        palindromes.sort()
        return palindromes


    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Sorting nums takes O(N log N)
        nums.sort()

        # Calculate prefix sums for optimized cost calculation
        # Takes O(N) time and O(N) space
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        # Define the cost calculation function using prefix sums
        # Takes O(log N) time using binary search
        def calculate_cost_optimized(y: int) -> int:
            # Find the index where elements are >= y using binary search
            # bisect_left finds the first index i such that a[i] >= x
            split_idx = bisect.bisect_left(nums, y)

            # Cost for elements nums[0]...nums[split_idx-1] (which are < y) is sum(y - nums[i])
            # = split_idx * y - sum(nums[0]...nums[split_idx-1])
            cost_left = split_idx * y - prefix_sum[split_idx]

            # Cost for elements nums[split_idx]...nums[n-1] (which are >= y) is sum(nums[i] - y)
            # = sum(nums[split_idx]...nums[n-1]) - (n - split_idx) * y
            cost_right = (prefix_sum[n] - prefix_sum[split_idx]) - (n - split_idx) * y

            return cost_left + cost_right


        # Find the median value. O(1) after sorting.
        median_idx = (n - 1) // 2
        median_val = nums[median_idx]

        # Generate all palindromes less than 10^9.
        # Number of palindromes < 10^9 is constant (around 10^5). Let P be this number.
        # Generation takes roughly O(P) iterations for first_half. String operations and int conversion are fast.
        # Sorting takes O(P log P). Total generation O(P log P).
        all_palindromes = self.generate_palindromes(10**9)

        # Find the palindromes closest to the median value. O(log P).
        # Use bisect_left to find the index of the first element >= median_val
        idx_left = bisect.bisect_left(all_palindromes, median_val)
        # Use bisect_right to find the index of the first element > median_val
        # The element at index idx_right - 1 is the largest element <= median_val
        idx_right = bisect.bisect_right(all_palindromes, median_val)

        candidate_palindromes_refined = set()

        # Smallest palindrome >= median_val is at index idx_left
        if idx_left < len(all_palindromes):
             candidate_palindromes_refined.add(all_palindromes[idx_left])

        # Largest palindrome <= median_val is at index idx_right - 1
        if idx_right > 0:
             candidate_palindromes_refined.add(all_palindromes[idx_right - 1])

        min_cost = float('inf')

        # Calculate cost for the few candidate palindromes (at most 2). O(K * log N) where K <= 2.
        for pal_candidate in candidate_palindromes_refined:
            current_cost = calculate_cost_optimized(pal_candidate)
            min_cost = min(min_cost, current_cost)

        return min_cost