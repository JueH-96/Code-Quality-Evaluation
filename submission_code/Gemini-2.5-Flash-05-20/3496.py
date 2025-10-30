import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        
        # Helper function to calculate the maximum height a single worker can reduce
        # given their workerTime and a maximum allowed time T.
        # The time taken for a worker with base time `w` to reduce height by `x` is
        # w * (1 + 2 + ... + x) = w * x * (x + 1) / 2.
        def get_max_dig_for_worker(worker_time: int, T: int) -> int:
            # We want to find the largest integer x such that:
            # worker_time * x * (x + 1) // 2 <= T
            #
            # Binary search for x:
            # The lower bound for x is 0.
            # The upper bound for x can be estimated. If worker_time = 1 (minimum) and
            # T = 6 * 10^15 (approximate maximum possible T), then:
            # x * (x + 1) // 2 <= 6 * 10^15
            # x^2 / 2 approx 6 * 10^15
            # x^2 approx 12 * 10^15
            # x approx sqrt(12 * 10^15) approx 3.46 * 10^7
            # So, an upper bound of 4 * 10^7 or 2 * 10^8 is safe for the binary search.
            low_x, high_x = 0, 2 * (10**8) 
            ans_x = 0
            
            while low_x <= high_x:
                mid_x = (low_x + high_x) // 2
                
                # Calculate time_needed for `mid_x` blocks.
                # Python's integers handle arbitrary precision, so intermediate overflow isn't an issue.
                # mid_x * (mid_x + 1) is always an even number, so // 2 results in an exact integer.
                time_needed = worker_time * mid_x * (mid_x + 1) // 2
                
                if time_needed <= T:
                    ans_x = mid_x  # `mid_x` is a possible amount of work, try for more
                    low_x = mid_x + 1
                else:
                    high_x = mid_x - 1 # `mid_x` is too much, try less
            return ans_x

        # Check function: determines if mountainHeight can be reduced to 0 within 'T' seconds.
        # This is done by summing up the maximum height each worker can reduce within time T.
        def check(T: int) -> bool:
            total_diggable_height = 0
            for w_time in workerTimes:
                total_diggable_height += get_max_dig_for_worker(w_time, T)
                # Optimization: if the total diggable height already meets or exceeds
                # mountainHeight, we don't need to check remaining workers.
                if total_diggable_height >= mountainHeight:
                    return True
            return total_diggable_height >= mountainHeight

        # Main binary search for the minimum time 'T'.
        # The range for T:
        # Minimum T: A single block costs at least min(workerTimes[i]) seconds. So 1 is a lower bound.
        low_T = 1
        # Maximum T: One worker (workerTimes[i]=1) digs all mountainHeight (10^5 blocks).
        # Time = 1 * 10^5 * (10^5 + 1) / 2 approx 5 * 10^9.
        # Max workerTimes[i] is 10^6. If mountainHeight = 10^5 and one workerTimes[i] = 10^6.
        # This could be 10^6 * 10^5 * (10^5 + 1) / 2 approx 5 * 10^15.
        # So, a safe upper bound for high_T can be set to 6 * 10^15.
        high_T = 6 * (10**15) 
        ans = high_T  # Initialize minimum time found so far to a very large value
        
        while low_T <= high_T:
            mid_T = (low_T + high_T) // 2
            
            if check(mid_T):
                # If `mid_T` is enough time, it's a possible answer. Try for a smaller time.
                ans = mid_T
                high_T = mid_T - 1
            else:
                # If `mid_T` is not enough, we need more time.
                low_T = mid_T + 1
                
        return ans