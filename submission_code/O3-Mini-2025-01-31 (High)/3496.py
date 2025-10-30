import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # We want to reduce the mountainHeight to 0.
        # Each worker i, with work time t, takes:
        #   t + 2*t + ... + x*t = t*(x*(x+1)//2) seconds
        # to reduce the mountain by x units.
        # So, given a total time T seconds, worker i can reduce at most x units such that:
        #   t * (x*(x+1)//2) <= T.
        # Multiply both sides by 2:
        #   t * x*(x+1) <= 2*T  ->  x*(x+1) <= (2*T) / t.
        #
        # We can solve for x:
        #   x = floor((-1 + sqrt(1 + 8*T/t)) / 2)
        #
        # Since workers work in parallel, T seconds is enough if the sum of maximum
        # reductions across all workers is at least mountainHeight.
        #
        # We use binary search for the minimum T.
        
        # Lower bound is 0 seconds.
        # An upper bound is the time needed when the fastest worker (min(workerTimes))
        # does all the work: t * (mountainHeight*(mountainHeight+1)//2).
        low = 0
        high = min(workerTimes) * (mountainHeight * (mountainHeight + 1) // 2)
        
        while low < high:
            mid = (low + high) // 2
            total = 0
            
            for t in workerTimes:
                # Calculate maximum reductions by worker with time t in mid seconds.
                # Solve: t*(x*(x+1)//2) <= mid  <=>  x*(x+1) <= 2*mid/t.
                # The quadratic formula gives: x = floor((-1 + sqrt(1+8*mid/t))/2)
                x = int((math.sqrt(1 + 8 * mid / t) - 1) // 2)
                total += x
                
                if total >= mountainHeight:  # Enough reduction reached
                    break
            
            if total >= mountainHeight:
                high = mid
            else:
                low = mid + 1
        
        return low