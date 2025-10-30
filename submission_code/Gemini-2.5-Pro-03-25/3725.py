import collections
from typing import List

# Helper function to calculate the count of valid subarrays for a given element
# This function can be defined outside the class or as a static method.
def calculate_count_static(m1: int, m2: int, k: int) -> int:
    """
    Calculates the number of pairs (x, y) such that:
    0 <= x < m1  (x represents the distance from the element i to the left endpoint L: x = i - L)
    0 <= y < m2  (y represents the distance from the element i to the right endpoint R: y = R - i)
    x + y <= k - 1 (derived from the subarray length constraint R - L + 1 <= k)

    This count represents the number of subarrays of length at most k where the element at index i 
    is the designated value (either max or min), considering its immediate larger/smaller neighbors.

    m1: Number of possible positions for the left endpoint L relative to i. (i - left_bound)
    m2: Number of possible positions for the right endpoint R relative to i. (right_bound - i)
    k: Maximum subarray length constraint.
    
    The calculation is based on summing the number of valid y for each valid x:
    Sum_{x=0 to m1-1} (number of y such that 0 <= y < m2 and y <= k - 1 - x)
    Sum_{x=0 to m1-1} max(0, min(m2, k - x))
    """
    if m1 <= 0 or m2 <= 0:
        # If either interval length is zero or negative, no subarrays are possible.
        return 0

    count = 0

    # The formula Sum_{x=0 to m1-1} max(0, min(m2, k - x)) is evaluated efficiently.
    # The term max(0, min(m2, k - x)) simplifies based on the value of k - x relative to m2 and 0.

    # Case 1: k - x >= m2  (implies x <= k - m2)
    # The term is max(0, m2) = m2 (since m2 >= 1)
    # We sum m2 for x from 0 up to min(m1 - 1, k - m2).
    end1 = min(m1 - 1, k - m2)
    if end1 >= 0:
        # Number of terms in this range [0, end1] is end1 + 1.
        num_terms1 = end1 + 1
        count += m2 * num_terms1

    # Case 2: 0 <= k - x < m2 (implies k - m2 < x < k)
    # The term is max(0, k - x) = k - x (since k - x >= 0)
    # We sum (k - x) for x from max(0, k - m2 + 1) up to min(m1 - 1, k - 1).
    # Note: if x >= k, then k - x <= 0, and max(0, k - x) = 0, so we only need to sum up to k - 1.
    start2 = max(0, k - m2 + 1)
    end2 = min(m1 - 1, k - 1) 
    
    if start2 <= end2:
        # Sum of an arithmetic progression: (k - start2), (k - start2 - 1), ..., (k - end2)
        num_terms2 = end2 - start2 + 1
        first_term = k - start2
        last_term = k - end2
        # The sum is (Number of terms) * (First term + Last term) / 2
        # Use integer division for safety, although the sum should be an integer.
        sum_range2 = num_terms2 * (first_term + last_term) // 2
        count += sum_range2

    # Case 3: k - x < 0 (implies x >= k)
    # The term is max(0, negative) = 0. This contributes 0 to the sum.
    # This range is implicitly handled as the summation for x only goes up to m1-1, 
    # and the calculation for Range 2 stops at x = k-1.

    return count

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of the maximum and minimum elements of all subarrays
        of `nums` with length at most `k`.

        Args:
            nums: The input integer array.
            k: The maximum length of the subarrays to consider.

        Returns:
            The total sum of min + max for all valid subarrays.
        """
        n = len(nums)
        
        # --- Calculate boundaries for maximum contribution ---
        # left[i]: index of the nearest element to the left of i such that nums[left[i]] > nums[i]
        stack = []
        left = [-1] * n
        for i in range(n):
            # Pop elements less than or equal to nums[i] to find the first element > nums[i]
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)

        # right[i]: index of the nearest element to the right of i such that nums[right[i]] >= nums[i]
        # Using >= ensures that for duplicate maximums, we assign the subarray to the leftmost occurrence.
        stack = []
        right = [n] * n
        for i in range(n - 1, -1, -1):
            # Pop elements strictly smaller than nums[i] to find the first element >= nums[i]
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                # The element at stack[-1] is >= nums[i]
                right[i] = stack[-1]
            stack.append(i)
            
        # --- Calculate boundaries for minimum contribution ---
        # left_prime[i]: index of the nearest element to the left of i such that nums[left_prime[i]] < nums[i]
        stack = []
        left_prime = [-1] * n
        for i in range(n):
            # Pop elements greater than or equal to nums[i] to find the first element < nums[i]
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                left_prime[i] = stack[-1]
            stack.append(i)

        # right_prime[i]: index of the nearest element to the right of i such that nums[right_prime[i]] <= nums[i]
        # Using <= ensures that for duplicate minimums, we assign the subarray to the leftmost occurrence.
        stack = []
        right_prime = [n] * n
        for i in range(n - 1, -1, -1):
            # Pop elements strictly greater than nums[i] to find the first element <= nums[i]
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                # The element at stack[-1] is <= nums[i]
                right_prime[i] = stack[-1]
            stack.append(i)

        total_sum = 0
        for i in range(n):
            # --- Contribution of nums[i] as the maximum element ---
            l, r = left[i], right[i]
            # The element nums[i] is the maximum in subarrays nums[L:R+1] where l < L <= i <= R < r.
            # m1 = number of choices for L = i - l
            # m2 = number of choices for R = r - i
            m1, m2 = i - l, r - i 
            # Calculate how many of these subarrays have length at most k.
            count_max = calculate_count_static(m1, m2, k)
            total_sum += nums[i] * count_max
            
            # --- Contribution of nums[i] as the minimum element ---
            lp, rp = left_prime[i], right_prime[i]
            # The element nums[i] is the minimum in subarrays nums[L:R+1] where lp < L <= i <= R < rp.
            # m1p = number of choices for L = i - lp
            # m2p = number of choices for R = rp - i
            m1p, m2p = i - lp, rp - i
            # Calculate how many of these subarrays have length at most k.
            count_min = calculate_count_static(m1p, m2p, k)
            total_sum += nums[i] * count_min
            
        # Python integers handle arbitrary size, so overflow is not an issue.
        return total_sum