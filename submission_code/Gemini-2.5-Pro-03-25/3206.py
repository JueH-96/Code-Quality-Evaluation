from typing import List

class Solution:
  """
  Solves the problem of finding intersection values between two arrays based on element presence.
  """
  def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
      """
      Calculates two values:
      1. The count of indices i such that nums1[i] is present in nums2.
      2. The count of indices j such that nums2[j] is present in nums1.

      Args:
          nums1: The first list of integers (0-indexed).
          nums2: The second list of integers (0-indexed).

      Returns:
          A list containing the two calculated counts in the specified order.
      """
      # Create sets from the input lists for efficient membership testing.
      # Checking if an element is in a set takes O(1) average time complexity.
      # Creating a set from a list of size k takes O(k) time.
      set1 = set(nums1)  # Contains unique elements from nums1
      set2 = set(nums2)  # Contains unique elements from nums2
      
      # Calculate count1: Iterate through nums1 and check if each element exists in set2.
      # The generator expression '(1 for num in nums1 if num in set2)' yields 1 
      # for each element `num` in `nums1` that is found in `set2`.
      # sum() efficiently aggregates these counts.
      # The time complexity for this step is O(n), where n is the length of nums1, 
      # because we iterate through nums1 once, and each lookup in set2 is O(1) on average.
      count1 = sum(1 for num in nums1 if num in set2)
      
      # Calculate count2: Iterate through nums2 and check if each element exists in set1.
      # Similar logic applies as for count1.
      # The time complexity for this step is O(m), where m is the length of nums2.
      count2 = sum(1 for num in nums2 if num in set1)
      
      # The overall time complexity is dominated by the set creations and the iterations:
      # O(n) + O(m) + O(n) + O(m) = O(n + m).
      # The overall space complexity is determined by the space needed for the sets:
      # O(k + l), where k is the number of unique elements in nums1 and l is the number
      # of unique elements in nums2. In the worst case, this is O(n + m).
      
      # Return the two calculated counts as a list of size 2.
      return [count1, count2]