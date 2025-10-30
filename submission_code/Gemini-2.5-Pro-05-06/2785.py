from typing import List

class Solution:
  def semiOrderedPermutation(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Constraints: 2 <= nums.length == n <= 50.
    # This means n >= 2, so 1 and n are distinct values.

    # Find the initial 0-indexed positions of 1 and n.
    pos1 = -1 # Stores the index of element 1
    posn = -1 # Stores the index of element n
    
    # Iterate through the list to find positions.
    # nums.index(value) could also be used, but a single pass is clear.
    for i in range(n):
      if nums[i] == 1:
        pos1 = i
      elif nums[i] == n: # Using elif because 1 and n are distinct given n >= 2.
        posn = i
      
      # Optimization: if both are found, can break early.
      # For n <= 50, this is minor. A full scan is simple.
      # if pos1 != -1 and posn != -1:
      #    break
        
    # Number of swaps to move element 1 from pos1 to index 0.
    # Each swap moves it one position to the left.
    swaps_for_1 = pos1
    
    # Number of swaps to move element n from posn to index n-1.
    # Each swap moves it one position to the right.
    # This is calculated based on its *original* position relative to n-1.
    swaps_for_n = (n - 1) - posn
    
    # The total number of swaps is initially the sum of swaps for 1 and swaps for n.
    # This sum counts the minimum swaps required if the movements of 1 and n
    # are considered as if they don't "help" each other.
    total_swaps = swaps_for_1 + swaps_for_n
    
    # Adjustment:
    # If element 1 is initially to the right of element n (i.e., pos1 > posn),
    # then for 1 to reach index 0, it must move leftwards past the original position of n.
    # Similarly, for n to reach index n-1, it must move rightwards past the original position of 1.
    # Schematically: ... n ... 1 ...
    # When 1 moves left and eventually swaps with n (e.g., [..., n, 1, ...] becomes [..., 1, n, ...]),
    # this single swap operation moves 1 one step left AND moves n one step right.
    # This specific swap contributes to both counts (swaps_for_1 and swaps_for_n).
    # Therefore, the sum `swaps_for_1 + swaps_for_n` has double-counted this one swap.
    # We must subtract 1 from the total to correct for this.
    #
    # If 1 is initially to the left of n (pos1 < posn), their paths do not cross
    # in this specific manner. 1 moves left, n moves right. No single swap involving 
    # both 1 and n directly contributes to both goals like this.
    # Note: pos1 cannot be equal to posn because elements 1 and n are distinct (since n >= 2).
    if pos1 > posn:
      total_swaps -= 1
      
    return total_swaps