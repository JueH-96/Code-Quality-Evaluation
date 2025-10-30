from typing import List

class Solution:
  def largestPerimeter(self, nums: List[int]) -> int:
    nums.sort() # Sorts in non-decreasing order: a_0, a_1, ..., a_{n-1}

    # current_sum_of_considered_sides stores the sum of elements a_0, ..., a_j
    # which are currently being considered for forming a polygon.
    # Initially, for j = n-1, it's the sum of all elements in nums.
    current_sum_of_considered_sides = sum(nums)
    n = len(nums)

    # We need at least 3 sides for a polygon.
    # The loop considers sets of sides {a_0, ..., a_j}.
    # a_j is the largest side in this set.
    # The number of sides is j+1. So, j+1 >= 3 implies j >= 2.
    # The loop iterates j from n-1 down to 2.
    # This corresponds to considering n sides, then n-1 sides, ..., down to 3 sides.
    # j is the index of the largest side in the current candidate set {a_0, ..., a_j}.
    for j in range(n - 1, 1, -1): 
      
      # At this point, current_sum_of_considered_sides is sum(a_0, ..., a_j).
      # The largest side in this set is a_j (which is nums[j]).
      longest_side = nums[j]
      # The sum of the other sides (a_0, ..., a_{j-1}) is 
      # current_sum_of_considered_sides - longest_side.
      sum_of_other_sides = current_sum_of_considered_sides - longest_side
      
      # Polygon condition: sum of shorter sides > longest side
      if sum_of_other_sides > longest_side:
        # If condition holds, {a_0, ..., a_j} can form a polygon.
        # Its perimeter is current_sum_of_considered_sides.
        # Since we iterate j downwards (considering subsets with more elements first,
        # and these elements are from the full sorted list starting from smallest),
        # this is the largest possible perimeter according to the greedy strategy.
        return current_sum_of_considered_sides
      else:
        # nums[j] is too large for the set {a_0, ..., a_j} to form a polygon.
        # Discard nums[j] (effectively a_j).
        # The sum for the next iteration (considering {a_0, ..., a_{j-1}}) will be
        # current_sum_of_considered_sides - nums[j].
        current_sum_of_considered_sides -= nums[j] 
    
    # If the loop completes, it means no set of k >= 3 sides of the form {a_0, ..., a_j}
    # could form a polygon. This implies no polygon can be formed at all.
    return -1