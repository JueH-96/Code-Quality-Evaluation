from typing import List

class Solution:
  def minimumSum(self, nums: List[int]) -> int:
    n = len(nums)
    min_total_sum = float('inf')

    # Constraints: 3 <= nums.length <= 50.
    # This means n is at least 3, so loops will be valid.

    # j is the index of the peak element nums[j].
    # For a triplet (i, j, k) with i < j < k:
    # Minimum j is 1 (when i=0, k=2).
    # Maximum j is n-2 (when i=n-3, k=n-1).
    # So, j ranges from 1 to n-2, inclusive.
    # Python's range(start, stop) goes up to stop-1.
    # So, range(1, n-1) covers j from 1 up to (n-1)-1 = n-2.
    for j in range(1, n - 1):
        current_nums_j = nums[j]
        
        min_left_val = float('inf')
        # Find the smallest nums[i] such that i < j and nums[i] < nums[j].
        # i ranges from 0 to j-1.
        for i in range(j):
            if nums[i] < current_nums_j:
                min_left_val = min(min_left_val, nums[i])
        
        # If no such left element exists (i.e., all nums[i] for i < j are >= nums[j]),
        # then nums[j] cannot be a peak of a mountain triplet.
        # Skip to the next potential peak j.
        if min_left_val == float('inf'):
            continue 
        
        min_right_val = float('inf')
        # Find the smallest nums[k] such that k > j and nums[k] < nums[j].
        # k ranges from j+1 to n-1.
        for k in range(j + 1, n):
            if nums[k] < current_nums_j:
                min_right_val = min(min_right_val, nums[k])
        
        # If no such right element exists (i.e., all nums[k] for k > j are >= nums[j]),
        # then nums[j] cannot be a peak of a mountain triplet with the chosen min_left_val.
        # Skip to the next potential peak j.
        if min_right_val == float('inf'):
            continue
        
        # If both a valid smallest left element (min_left_val) and 
        # a valid smallest right element (min_right_val) were found,
        # they form a mountain triplet with nums[j] as the peak.
        # Calculate the sum and update the overall minimum sum.
        current_sum = min_left_val + current_nums_j + min_right_val
        min_total_sum = min(min_total_sum, current_sum)

    # After checking all possible peaks j:
    # If min_total_sum is still float('inf'), no mountain triplet was found.
    if min_total_sum == float('inf'):
        return -1
    else:
        # The sum is composed of integers from nums, so it's an integer.
        # float('inf') is a float, so min_total_sum could be a float if initialized to it.
        # Cast to int for the return type.
        return int(min_total_sum)