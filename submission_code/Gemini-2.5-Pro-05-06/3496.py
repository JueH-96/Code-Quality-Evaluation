import math
from typing import List

class Solution:
  def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
    
    # check(current_T) returns True if it's possible to reduce mountainHeight to 0 within current_T seconds.
    # False otherwise.
    # Parameters for mountainHeight and workerTimes are taken from the outer scope.
    def check(current_T: int) -> bool:
      total_h_reduced = 0
      for w_time in workerTimes:
        # workerTimes[i] >= 1 per constraints, so w_time > 0 always.

        # Max height x this worker can reduce is such that: w_time * x * (x+1) / 2 <= current_T
        # This means: x * (x+1) <= 2 * current_T / w_time
        # Let val_int_div = floor(2 * current_T / w_time)
        
        val_int_div = (2 * current_T) // w_time # Result is non-negative as current_T >=0, w_time >=1.

        # We need to find max integer x such that x^2 + x - val_int_div <= 0
        # The positive root of x^2 + x - val_int_div = 0 is x_root = (-1 + sqrt(1 + 4*val_int_div)) / 2
        # So, x_can_reduce = floor(x_root)
        
        # Calculate 1 + 4*val_int_div. This will be >= 1.
        isqrt_arg = 1 + 4 * val_int_div
        # k = floor(sqrt(isqrt_arg)). This k will be >= 1.
        k = math.isqrt(isqrt_arg)
        # x_can_reduce = floor((k-1)/2). This will be >= 0.
        # This formula correctly gives x_can_reduce = 0 if val_int_div is 0 or 1 
        # (i.e., worker cannot reduce by 1 unit of height, which requires val_int_div >= 2).
        x_can_reduce = (k - 1) // 2
        
        total_h_reduced += x_can_reduce
        
        # Optimization: if total height reduced already meets requirement, no need to check other workers.
        if total_h_reduced >= mountainHeight:
          return True
          
      return total_h_reduced >= mountainHeight

    # Binary search for the minimum time T.
    low = 0 # Minimum possible time.
    
    # Upper bound for T: worst case is 1 worker with max workerTime factor (10^6), 
    # reducing max mountainHeight (10^5).
    # Time = worker_time_factor * H * (H+1) / 2
    # max_H_val_const = 10^5 
    # max_W_val_const = 10^6 
    # high_initial = max_W_val_const * (max_H_val_const * (max_H_val_const + 1) // 2)
    # which is 10^6 * (10^5 * (10^5+1) // 2) = 10^6 * 5000050000 = 5000050000000000 (5.00005 * 10^15)
    high = 5000050000000000 
    
    min_total_seconds_ans = high # Initialize with the max possible time

    # Standard binary search for minimum value that satisfies condition check(T) is true.
    while low <= high:
      mid_T = low + (high - low) // 2 # Prevents overflow for low+high (not an issue in Python)
      
      if check(mid_T):
        min_total_seconds_ans = mid_T
        high = mid_T - 1 # Condition met, try to find an even smaller T
      else:
        low = mid_T + 1 # mid_T is not enough, T must be larger
        
    return min_total_seconds_ans