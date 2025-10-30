from typing import List

class Solution:
  def minOperations(self, nums: List[int]) -> int:
    """
    Calculates the minimum number of operations to make all elements in a binary array equal to 1.
    An operation consists of choosing 3 consecutive elements and flipping them (0 to 1, 1 to 0).

    Args:
      nums: The input binary array. This list is modified in-place during the execution.

    Returns:
      The minimum number of operations required to make all elements 1. If it's impossible, returns -1.
    """
    n = len(nums)
    operations = 0
    
    # We iterate through the array using a greedy approach.
    # The iteration goes from index 0 up to n-3. This is because the operation
    # involves 3 consecutive elements: i, i+1, i+2. The last possible operation
    # starts at index n-3, affecting elements n-3, n-2, and n-1.
    for i in range(n - 2):
      # Check the element at the current index i.
      # If nums[i] is 0, we must perform an operation starting at index i
      # to flip nums[i] to 1. This is necessary because any operation starting
      # at an index j > i will not affect nums[i]. The operations starting
      # before i have already been decided based on the state of elements < i.
      # The greedy choice is forced: if nums[i] is 0, we must flip starting at i.
      if nums[i] == 0:
        # Perform the flip operation on the triplet nums[i], nums[i+1], nums[i+2].
        # Flipping a bit can be done efficiently using the XOR operator with 1.
        # 0 ^ 1 = 1
        # 1 ^ 1 = 0
        nums[i] ^= 1  # Flip nums[i]
        nums[i+1] ^= 1 # Flip nums[i+1]
        nums[i+2] ^= 1 # Flip nums[i+2]
        
        # Increment the counter for the number of operations performed.
        operations += 1
    
    # After the loop completes, elements from index 0 to n-3 have been processed.
    # The greedy strategy ensures that if it's possible to make these elements 1,
    # they will be 1 at this point.
    # Now, we need to check the last two elements of the array: nums[n-2] and nums[n-1].
    # These elements could have been affected by operations starting at n-4 and n-3.
    # If either nums[n-2] or nums[n-1] is 0 at this point, it means it's impossible
    # to make the entire array equal to 1s. There are no further valid operations
    # that can flip these elements without affecting elements outside the array bounds
    # or disturbing the already fixed prefix.
    if nums[n-2] == 0 or nums[n-1] == 0:
      # If either of the last two elements is 0, return -1 indicating impossibility.
      return -1
    else:
      # If both nums[n-2] and nums[n-1] are 1, it implies that all elements
      # in the array are now 1. Return the total count of operations performed.
      return operations