from typing import List

class Solution:
  def countWays(self, nums: List[int]) -> int:
    nums.sort()
    n = len(nums)
    ans = 0

    # Case: 0 students selected (k=0 in the problem statement's notation)
    # All n students are not selected.
    # For any student i (who is not selected) to be happy, 0 < nums[i].
    # This condition must hold for all students.
    # Since nums is sorted, this is equivalent to checking nums[0] (the smallest value).
    # If nums[0] > 0, then all nums[i] > 0, so all students are happy.
    if nums[0] > 0:
      ans += 1
    
    # Case: n students selected (k=n in the problem statement's notation)
    # All n students are selected.
    # For any student i (who is selected) to be happy, n > nums[i].
    # This condition must hold for all students.
    # Since nums is sorted, this is equivalent to checking nums[n-1] (the largest value).
    # The problem constraints state 0 <= nums[i] < nums.length (which is n).
    # So, nums[i] <= n-1 for all i.
    # Thus, nums[n-1] <= n-1. This implies nums[n-1] < n is always true.
    # Therefore, selecting all n students is always a valid way.
    # This `k=n` way is distinct from `k=0` way since n >= 1 (problem constraint).
    ans += 1
    
    # Case: k_sel students selected, where 0 < k_sel < n
    # (k_sel corresponds to k in the problem statement's notation).
    # The students selected are those with the k_sel smallest nums values.
    # After sorting, these are students whose values are nums[0]...nums[k_sel-1].
    # For these selected students to be happy: k_sel > nums[i] for all i in [0, k_sel-1].
    # The critical condition (for the "hardest to please" selected student) is k_sel > nums[k_sel-1].
    
    # The remaining (n - k_sel) students are not selected.
    # After sorting, these are students whose values are nums[k_sel]...nums[n-1].
    # For these non-selected students to be happy: k_sel < nums[i] for all i in [k_sel, n-1].
    # The critical condition (for the "easiest to displease" non-selected student) is k_sel < nums[k_sel].
    
    # Both conditions must hold for k_sel to be a valid number of selections.
    for k_sel in range(1, n): # k_sel iterates from 1 up to (but not including) n
      if nums[k_sel-1] < k_sel and nums[k_sel] > k_sel:
        ans += 1
        
    return ans