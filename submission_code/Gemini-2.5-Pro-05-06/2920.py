import collections
from typing import List

class Solution:
  def minimumSeconds(self, nums: List[int]) -> int:
    n = len(nums)
    
    # According to constraints, 1 <= n.
    # If n=1, the array is already equalized, so 0 seconds are needed.
    # This case is handled correctly by the main logic:
    # e.g. nums = [100]. positions will be {100: [0]}.
    # For val_target=100: current_indices=[0], m=1.
    # current_max_gap_len becomes n (which is 1).
    # seconds_for_val_target = 1 // 2 = 0.
    # min_total_seconds is initialized to n=1, then updated to min(1, 0) = 0.
    # Correctly returns 0.
    
    # Store indices for each number.
    # Example: if nums = [1, 2, 1, 2], n = 4
    # positions will be {1: [0, 2], 2: [1, 3]}
    positions = collections.defaultdict(list)
    for i, num_val in enumerate(nums):
        positions[num_val].append(i)
        
    # Initialize min_total_seconds.
    # The maximum possible seconds is n // 2 (e.g., for nums = [X, Y, Y, ..., Y] where X != Y).
    # Initializing with n is safe; n // 2 would also be a valid (tighter) upper bound.
    min_total_seconds = n 
    
    # Iterate over each unique number present in the input array.
    # For each unique number `val_target`, calculate the time it would take
    # to make all elements in the array equal to `val_target`.
    for val_target in positions:
        current_indices = positions[val_target] # List of 0-indexed positions where `val_target` occurs.
                                                # These are sorted, e.g. [idx_0, idx_1, ..., idx_{m-1}].
        m = len(current_indices)
        
        # `current_max_gap_len` will store the length of the longest circular segment
        # defined by two consecutive occurrences of `val_target`.
        # This length is the distance between these two occurrences.
        current_max_gap_len = 0
        if m == 1:
            # If `val_target` appears only once (at current_indices[0]), it needs to spread across the entire array.
            # The "gap" is from this single occurrence, around the circle, back to itself.
            # The length of this gap (distance) is `n`.
            current_max_gap_len = n
        else:
            # If `val_target` appears multiple times, calculate the maximum gap length between
            # consecutive occurrences.
            # The first gap is from current_indices[0] to current_indices[1].
            # The second gap is from current_indices[1] to current_indices[2].
            # ...
            # The last gap is from current_indices[m-1] back to current_indices[0] (circularly).
            for i in range(m):
                idx1 = current_indices[i]
                # Get the next occurrence's index.
                # If i is m-1 (last element in current_indices), (i+1)%m is 0,
                # so idx2 becomes current_indices[0] (the first occurrence), handling the wrap-around.
                idx2 = current_indices[(i + 1) % m]
                
                # Calculate distance (length of the gap) going from idx1 to idx2 in circular fashion.
                # Example: n=5, idx1=4, idx2=1. Path from 4 is 4 -> 0 -> 1. Gap length is 2.
                # Formula: (1 - 4 + 5) % 5 = 2.
                # Example: n=5, idx1=1, idx2=4. Path from 1 is 1 -> 2 -> 3 -> 4. Gap length is 3.
                # Formula: (4 - 1 + 5) % 5 = 3.
                gap_len = (idx2 - idx1 + n) % n
                
                current_max_gap_len = max(current_max_gap_len, gap_len)
        
        # The time needed for this `val_target` to fill its largest gap (of length `current_max_gap_len`)
        # is floor(current_max_gap_len / 2).
        # Example: If a gap is X . . . X (gap_len=4, e.g., indices 0,1,2,3 with X at 0 and 4 if n=5 or more),
        # it implies elements at indices 1, 2, 3 need to become X.
        # The element at index 2 (middle) is furthest, requiring 2 seconds (4 // 2).
        # Example: If a gap is X . . . . X (gap_len=5),
        # elements at indices 2 and 3 are "middle-most", requiring 2 seconds (5 // 2).
        seconds_for_val_target = current_max_gap_len // 2
        
        # Update the overall minimum seconds found so far.
        min_total_seconds = min(min_total_seconds, seconds_for_val_target)
            
    return min_total_seconds