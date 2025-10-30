from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:

        def can_reduce_height(t: int, h: int, wt_list: List[int]) -> bool:
            """
            Checks if the mountain of height h can be reduced to 0
            within time t by workers with times wt_list.
            """
            total_reduced = 0
            for wt in wt_list:
                # For a given worker wt and time t, find the maximum height reduction x.
                # Time taken for reduction x is wt * x * (x + 1) / 2.
                # We need the largest integer x >= 0 such that wt * x * (x + 1) / 2 <= t.
                # This is equivalent to wt * x * (x + 1) <= 2 * t.
                # Since wt >= 1, this is equivalent to x * (x + 1) <= floor((2 * t) / wt).
                # floor((2 * t) / wt) is (2 * t) // wt using integer division.
                
                # Calculate the effective limit for x*(x+1)
                # Use 2 * t first to potentially get a larger number before integer division
                # by wt, which helps maintain precision compared to (2 * t / wt) floor.
                # This division is safe because wt >= 1.
                limit_val = (2 * t) // wt 

                # We need to find the largest integer x >= 0 such that x * (x + 1) <= limit_val.
                # The function f(x) = x * (x + 1) is strictly increasing for x >= 0.
                # We can use binary search to find this largest x in the range [0, h].
                # A worker never needs to reduce height more than the total height h to satisfy the condition >=h.
                # The maximum useful height reduction by a single worker is h.
                
                low_x, high_x = 0, h # Search range for height reduction x: [0, h] inclusive
                current_max_x = 0 # Stores the largest x found so far that satisfies the condition

                # Binary search for the maximum x in [0, h]
                while low_x <= high_x:
                    m_x = low_x + (high_x - low_x) // 2
                    
                    # Check the condition m_x * (m_x + 1) <= limit_val.
                    # m_x is at most h = 10^5. m_x * (m_x + 1) is approx 10^{10}.
                    # limit_val is approx 4*10^16.
                    # Both numbers fit in Python's arbitrary precision integers.
                    
                    # Calculate the left hand side of the condition.
                    # In Python 3, this is safe from overflow for the given constraints.
                    lhs = m_x * (m_x + 1)
                    
                    # Check if m_x * (m_x + 1) <= limit_val.
                    # If lhs <= limit_val, it means reducing by m_x is possible within the time t for this worker.
                    # We update current_max_x and try to find a larger possible reduction.
                    
                    if lhs <= limit_val:
                        # Reduction m_x is achievable. m_x is a candidate for the maximum.
                        current_max_x = m_x
                        low_x = m_x + 1 # Try larger reductions by searching in the upper half
                    else:
                        # Reduction m_x is too large for time t. Need to reduce less.
                        high_x = m_x - 1 # Try smaller reductions by searching in the lower half
                
                # After the inner binary search, current_max_x is the maximum height this worker can reduce within time t.
                total_reduced += current_max_x

            # Check if the total height reduced by all workers is sufficient.
            return total_reduced >= h

        # Main binary search for the minimum time t.
        # The time t must be non-negative. Lower bound is 0.
        low = 0
        
        # Determine a safe upper bound for the binary search.
        # Maximum possible time is when one worker with the maximum workerTime reduces the entire mountainHeight.
        # Max wt = 10^6, Max h = 10^5. Time = wt * h * (h + 1) / 2.
        # Approx max time = 10^6 * 10^5 * 10^5 / 2 = 0.5 * 10^16.
        # A generous upper bound like 2 * 10^16 + 7 is safe.
        # This bound is larger than any possible answer based on the constraints.
        high = 2 * 10**16 + 7 

        # Initialize the answer to the upper bound, which is guaranteed to be achievable.
        ans = high

        # Binary search loop for the minimum time t
        while low <= high:
            mid = low + (high - low) // 2
            
            if can_reduce_height(mid, mountainHeight, workerTimes):
                # If it's possible to reduce height to 0 within `mid` seconds,
                # `mid` is a possible answer. We try to find a smaller time.
                # Store `mid` as the current best answer and search in the lower half [low, mid - 1].
                ans = mid
                high = mid - 1
            else:
                # If it's not possible within `mid` seconds, we need more time.
                # Search in the upper half [mid + 1, high].
                low = mid + 1
        
        # The loop finishes when low > high. The last value of `mid` for which `can_reduce_height`
        # returned True is the minimum time required. This is stored in `ans`.
        return ans