import bisect
from typing import List

class Solution:
    # A class attribute to cache the list of palindromes. This ensures the generation
    # runs only once across multiple test cases.
    PALS = []

    def minimumCost(self, nums: List[int]) -> int:
        """
        Calculates the minimum cost to make all elements in an array equal to a single palindromic number.
        """
        
        # Generate and cache the list of palindromic numbers if not already done.
        # This list includes all palindromes up to 10^9.
        if not Solution.PALS:
            pals_set = set()
            # Generate odd-length palindromes (e.g., 121, 1331)
            for i in range(1, 100000):
                s = str(i)
                pals_set.add(int(s + s[:-1][::-1]))
            # Generate even-length palindromes (e.g., 11, 1221)
            for i in range(1, 10000):
                s = str(i)
                pals_set.add(int(s + s[::-1]))
            Solution.PALS = sorted(list(pals_set))

        n = len(nums)
        nums.sort()
        
        # The cost function sum(|x - y|) is minimized when y is the median.
        median = nums[n // 2]
        
        # Find the two palindromes closest to the median using binary search.
        idx = bisect.bisect_left(Solution.PALS, median)
        
        # Candidate 1: Largest palindrome less than or equal to the median.
        cost1 = float('inf')
        if idx > 0:
            p_smaller = Solution.PALS[idx - 1]
            cost1 = sum(abs(num - p_smaller) for num in nums)
        
        # Candidate 2: Smallest palindrome greater than or equal to the median.
        cost2 = float('inf')
        if idx < len(Solution.PALS):
            p_greater = Solution.PALS[idx]
            cost2 = sum(abs(num - p_greater) for num in nums)
            
        return min(cost1, cost2)