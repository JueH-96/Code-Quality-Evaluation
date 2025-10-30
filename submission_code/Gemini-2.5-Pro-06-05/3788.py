from typing import List

class Solution:
  def maxSum(self, nums: List[int]) -> int:
    """
    Calculates the maximum sum of a unique-element subarray that can be obtained
    after deleting elements from the input array.
    """
    
    # Step 1: Get the set of unique numbers from the input list.
    # This efficiently filters out any duplicate values.
    unique_nums = set(nums)
    
    # Step 2: Create a list of all unique non-negative numbers.
    non_negatives = [n for n in unique_nums if n >= 0]
    
    # Step 3: Check if there are any non-negative unique numbers.
    if non_negatives:
      # If yes, the maximum sum is the sum of all unique non-negative numbers.
      # This is because any negative number would decrease the sum.
      return sum(non_negatives)
    else:
      # If no, all unique numbers are negative. The problem requires a non-empty
      # selection. To maximize the sum, we must pick the largest
      # (least negative) unique number.
      return max(unique_nums)