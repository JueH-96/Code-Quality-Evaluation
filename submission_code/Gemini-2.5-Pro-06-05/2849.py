from typing import List

class Solution:
  def sumImbalanceNumbers(self, nums: List[int]) -> int:
    """
    Calculates the sum of imbalance numbers of all subarrays of nums.

    The solution iterates through all possible subarrays and calculates their
    imbalance numbers efficiently. For each starting index `i`, we extend
    the subarray one element at a time (from `j=i` to `n-1`).
    We maintain the imbalance of the current subarray `nums[i:j]` and
    update it incrementally when adding `nums[j]`.

    The imbalance of a set of unique numbers `S` can be calculated as:
    imbalance(S) = |S| - 1 - (number of adjacent pairs in S) for |S| > 1.

    When a new element `x` is added to the set `S`:
    - The number of unique elements `|S|` increases by 1, which adds 1 to the imbalance.
    - If `x-1` is already in `S`, a new adjacent pair is formed, which reduces the imbalance by 1.
    - If `x+1` is already in `S`, another adjacent pair is formed, also reducing the imbalance by 1.
    The total change in imbalance is `1 - (1 if x-1 in S) - (1 if x+1 in S)`.

    This O(n^2) approach is efficient enough for the given constraints.
    """
    n = len(nums)
    total_imbalance = 0

    for i in range(n):
      # For each subarray starting at index i
      seen = set()
      current_imbalance = 0
      
      for j in range(i, n):
        # We are considering the subarray nums[i:j+1]
        x = nums[j]
        
        if x in seen:
          # If x is already in the set of unique elements, the
          # imbalance number for nums[i:j+1] is the same as for nums[i:j].
          pass
        else:
          # Handle the first element of the subarray. Its imbalance is 0.
          if not seen:
            pass
          else:
            # x is a new unique element.
            # Update the imbalance based on its neighbors.
            delta = 1
            if (x - 1) in seen:
              delta -= 1
            if (x + 1) in seen:
              delta -= 1
            
            current_imbalance += delta
          
          seen.add(x)
        
        # Add the imbalance of the current subarray nums[i:j+1] to the total.
        total_imbalance += current_imbalance
            
    return total_imbalance