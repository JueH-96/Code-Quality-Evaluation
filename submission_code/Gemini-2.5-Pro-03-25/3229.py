from typing import List
import math # math is not used, can be removed. Kept for safety in case some standard math function was implicitly needed.

class Solution:
    """
    Finds the minimum cost to make all elements in an array equal to a single palindromic number.
    The cost is the sum of absolute differences between original elements and the target palindrome.
    """

    def _is_palindrome(self, k: int) -> bool:
        """
        Checks if a positive integer k is a palindrome.
        Helper method for minimumCost.
        """
        # Palindromic numbers must be positive as per problem description.
        if k <= 0: 
            return False
        
        # Convert the number to string to check for palindrome property.
        s = str(k)
        # A string is a palindrome if it reads the same forwards and backwards.
        return s == s[::-1]

    def _find_prev_palindrome(self, m: int) -> int:
        """
        Finds the largest palindromic number less than or equal to m.
        Helper method for minimumCost.
        """
        k = m
        while True:
            # Check if the current number k is a palindrome.
            if self._is_palindrome(k): 
                return k # Found the largest palindrome <= m
            k -= 1
            # The loop starts from m and decrements k. Since input constraints ensure nums[i] >= 1, 
            # the median m >= 1. The number 1 is a palindrome. 
            # So the loop is guaranteed to find a positive palindrome (at least 1) and terminate.

    def _find_next_palindrome(self, m: int) -> int:
        """
        Finds the smallest palindromic number greater than or equal to m.
        Helper method for minimumCost.
        """
        k = m
        while True:
            # Check if the current number k is a palindrome.
            if self._is_palindrome(k): 
                return k # Found the smallest palindrome >= m
            k += 1
            # The loop starts from m and increments k. Palindromes exist indefinitely 
            # (e.g., 1, 2, ..., 9, 11, 22, ..., 101, 111, ...), so this loop will always find a palindrome and terminate.
            # Python's arbitrary precision integers handle potential large values of k.

    def minimumCost(self, nums: List[int]) -> int:
        """
        Calculates the minimum cost to make the array nums equalindromic.
        An array is equalindromic if all its elements are equal to some positive palindromic number y.
        The cost is the sum of absolute differences |nums[i] - y|.
        """
        n = len(nums)
        # Sort the array. This is necessary to find the median efficiently.
        # The median property is key to solving this problem efficiently.
        nums.sort()
        
        # The median of the array minimizes the sum of absolute differences sum(|nums[i] - y|) over all integers y.
        # For odd n, the median is the unique middle element nums[n // 2].
        # For even n, any value in the interval [nums[n // 2 - 1], nums[n // 2]] is a median.
        # We use nums[n // 2] as our reference median point. This choice works correctly for both odd and even n cases.
        # If n is even, nums[n // 2] is the larger of the two middle elements.
        median = nums[n // 2]
        
        # Since the target value y must be a palindrome, the optimal y might not be exactly the median.
        # However, the cost function C(y) = sum(|nums[i] - y|) is convex. This implies that the minimum value
        # of C(y) for palindromic y must occur at a palindrome close to the median (where the minimum of C(y) occurs).
        # Specifically, the optimal palindromic y must be one of the two palindromes closest to the median:
        # p1 = the largest palindrome less than or equal to the median
        # p2 = the smallest palindrome greater than or equal to the median
        p1 = self._find_prev_palindrome(median) 
        p2 = self._find_next_palindrome(median) 
        
        # Calculate the total cost if we choose p1 as the target palindrome.
        # This involves summing the absolute differences between each element in nums and p1.
        cost1 = sum(abs(x - p1) for x in nums)
        
        # Optimization: If p1 and p2 are the same, it means the median itself is a palindrome.
        # In this special case, p1 (which equals p2) is the optimal palindromic target because it is the median.
        # The cost associated with the median is the minimum possible cost overall, so cost1 is the answer.
        # We can return early without calculating cost2.
        if p1 == p2:
            return cost1
            
        # If p1 and p2 are different palindromes, we need to calculate the cost for p2 as well.
        # This involves summing the absolute differences between each element in nums and p2.
        cost2 = sum(abs(x - p2) for x in nums)
            
        # The minimum cost required to make the array equalindromic is the smaller of the two calculated costs.
        # We choose the palindrome (p1 or p2) that results in a lower total cost.
        return min(cost1, cost2)