import math
from typing import List

class Solution:
  """
  Solves the minimum array length problem.
  Finds the minimum possible length of the array `nums` after performing
  the described operation any number of times.
  The operation involves selecting two distinct positive elements nums[i] and nums[j],
  removing them, and appending nums[i] % nums[j] to the array. Each operation reduces the array length by 1.
  The process stops when fewer than two positive numbers are available.
  """
  def minimumArrayLength(self, nums: List[int]) -> int:
    """
    Calculates the minimum possible length of the array.

    Args:
      nums: A list of positive integers.

    Returns:
      An integer denoting the minimum possible length of the array.
    """
    
    # Find the minimum element in the array.
    # Python's min() function is efficient, taking O(N) time.
    min_val = min(nums)
            
    # Check if any element `a` exists such that `a % min_val > 0`.
    # If such an element exists, it means that `min_val` does not divide all elements in `nums`.
    # This implies the greatest common divisor (GCD) `g` of all elements is strictly less than `min_val`.
    # The allowed operation `a % b` is similar to steps in the Euclidean algorithm.
    # It can be shown that we can eventually generate any number that is a multiple of the GCD `g`.
    # In particular, we can generate `g` itself.
    # Since `g < min_val`, we can generate a positive number smaller than the initial minimum.
    # If the GCD `g` is 1, we can eventually generate 1.
    # If 1 is present in the array (either initially or generated), we can use operations effectively:
    #   - Select `a > 1` and `1`. Replace with `a % 1 = 0`. This removes two positive numbers (`a`, `1`) and adds a `0`.
    #   - Select `1` and `1`. Replace with `1 % 1 = 0`. This removes two `1`s and adds a `0`.
    # Both operations reduce the count of positive numbers by 2.
    # This allows reducing the array until at most one positive number (which must be 1) remains.
    # Further analysis suggests that if we can generate a number smaller than the initial minimum 
    # (which happens if and only if `min_val` does not divide all elements), 
    # the minimum achievable length is 1. The final array could be `[0]` or `[1]`.

    for x in nums:
        # Check if min_val divides x. If not, x % min_val > 0.
        if x % min_val > 0:
             # If `min_val` does not divide all elements, minimum length is 1.
            return 1
            
    # If the loop completes without returning, it means `min_val` divides all elements in `nums`.
    # In this case, `min_val` must be the GCD of all elements (`g = min_val`).
    # We cannot generate any positive value smaller than `min_val` because any result `a % b` 
    # where `a` and `b` are multiples of `min_val` will also be a multiple of `min_val` (or 0).
    # The minimum positive value in the array will always be `min_val`.
    # The process stops when there is at most one positive element left, which must be `min_val`.
    # Operations involving `min_val`:
    #   - `(a, min_val) -> a % min_val = 0` (since `a` is a multiple of `min_val`). This consumes one `min_val`.
    #   - `(min_val, min_val) -> min_val % min_val = 0`. This consumes two `min_val`.
    # We want to minimize the final length. The minimum length is achieved by pairing up the `min_val` elements as much as possible.
    # Let `k` be the count of `min_val` in the initial array `nums`.
    # We can perform `floor(k/2)` operations of type `(min_val, min_val) -> 0`.
    # If `k` is even, all `min_val` elements can be paired up and eventually replaced by 0s. The final state has 0 positive elements.
    # If `k` is odd, one `min_val` element will remain after pairing up as many as possible. The final state has 1 positive element (`min_val`).
    # The number of elements required in the final state relates to the number of pairs.
    # The minimum length is `ceil(k / 2)`.
    
    # Count the occurrences of min_val in nums. O(N) time.
    count_min = nums.count(min_val) 
    
    # Calculate ceil(k / 2) using integer arithmetic.
    # For a positive integer k, ceil(k / 2) can be computed as (k + 1) // 2.
    min_len = (count_min + 1) // 2
    return min_len