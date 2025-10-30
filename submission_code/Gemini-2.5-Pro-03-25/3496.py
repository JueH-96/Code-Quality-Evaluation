import math
from typing import List

class Solution:
    """
    Solves the minimum time problem for mountain height reduction by workers.
    The problem asks for the minimum time T such that all workers, working simultaneously,
    can reduce the mountain height to 0. Each worker `i` takes `workerTimes[i] * x * (x+1) / 2`
    seconds to reduce the height by `x`. The total time is determined by the worker who takes the longest.

    The approach uses binary search on the possible minimum time `T`. For a given candidate time `T`,
    we check if it's possible for the workers to collectively reduce the mountain height by at least `mountainHeight`.
    To do this check efficiently, for each worker `i`, we calculate the maximum height reduction `x_i` they can achieve
    within time `T`. The total achievable reduction is the sum of all `x_i`. If this sum is >= `mountainHeight`,
    then time `T` is feasible.

    The maximum height reduction `x_i` for worker `i` within time `T` is the largest integer `x` such that:
    `workerTimes[i] * x * (x + 1) // 2 <= T`.
    This inequality can be rewritten as `x * (x + 1) <= (2 * T) // workerTimes[i]`.
    Let `K = (2 * T) // workerTimes[i]`. We need the largest integer `x` such that `x * (x + 1) <= K`.
    This can be found efficiently using the integer square root `math.isqrt`.
    Let `approx_x = math.isqrt(K)`. The actual maximum `x` is either `approx_x` or `approx_x - 1`.
    We check `approx_x * (approx_x + 1)`. If it's `<= K`, then `max_x = approx_x`. Otherwise, `max_x = approx_x - 1`.
    Python's arbitrary precision integers handle potentially large intermediate values.
    """
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        """
        Calculates the minimum number of seconds required for workers to reduce
        the mountain height to 0.

        Args:
            mountainHeight: The initial height of the mountain. Constraints: 1 <= mountainHeight <= 10^5.
            workerTimes: A list where workerTimes[i] is the base time for worker i. 
                         Constraints: 1 <= workerTimes.length <= 10^4, 1 <= workerTimes[i] <= 10^6.

        Returns:
            The minimum number of seconds required.
        """

        # Check base case constraint mountainHeight >= 1. If it could be 0, return 0.
        # Per constraints, this function will always be called with mountainHeight >= 1.
        # if mountainHeight <= 0: # This line is technically redundant due to constraints.
        #     return 0
        
        # Per constraints, workerTimes.length >= 1, so list is not empty.
        # Per constraints, workerTimes[i] >= 1.

        # Find the maximum worker time. This is needed to compute a safe upper bound for binary search.
        max_W = 0
        for w in workerTimes:
             if w > max_W:
                 max_W = w
        # Since workerTimes is non-empty and all elements are >= 1, max_W >= 1.
        
        H = mountainHeight
        # Calculate a safe upper bound for the binary search on time T.
        # The maximum time could theoretically be when the slowest worker does all the work alone.
        # Time = max_W * H * (H + 1) // 2. Python handles large integers automatically.
        # E.g., max_W=10^6, H=10^5 -> T_upper approx 5 * 10^15.
        T_upper = max_W * H * (H + 1) // 2

        # Define check function to determine if a given time T is sufficient.
        def check(T: int) -> bool:
            """Checks if time T allows workers to collectively reduce mountain height by at least mountainHeight."""
            total_reduction = 0
            for W in workerTimes:
                # Calculate max height reduction 'x' possible by this worker within time T.
                # We need max integer x >= 0 such that: W * x * (x + 1) // 2 <= T
                
                # Defensive check for W <= 0, though constraints guarantee W >= 1.
                if W <= 0: continue 

                # Use integer arithmetic to avoid floating point issues: x * (x + 1) <= (2 * T) // W
                K = (2 * T) // W
                
                # Since T >= 0 and W >= 1, K is guaranteed to be non-negative.
                if K < 0: continue # Failsafe check

                # Find the largest integer x >= 0 such that x * (x + 1) <= K.
                # Use integer square root `math.isqrt` for an efficient estimate.
                # approx_x = floor(sqrt(K))
                approx_x = math.isqrt(K)
                
                # Check if approx_x satisfies the condition x * (x + 1) <= K.
                if approx_x * (approx_x + 1) <= K:
                    # If approx_x works, it's the largest integer solution because
                    # (approx_x+1)^2 > K, which implies (approx_x+1)*(approx_x+2) > K.
                    max_xi = approx_x
                else:
                    # If approx_x * (approx_x + 1) > K, then approx_x is too large.
                    # The largest integer solution must be approx_x - 1.
                    # This holds because (approx_x - 1) * approx_x < approx_x^2 <= K (for approx_x > 0).
                    # The case approx_x = 0 is handled by the 'if' branch (0*1 <= K), so if we reach
                    # this 'else' branch, approx_x > 0 is guaranteed.
                    max_xi = approx_x - 1
                
                total_reduction += max_xi
                
                # Optimization: If the cumulative reduction already meets the target,
                # we can stop checking remaining workers for this T.
                if total_reduction >= mountainHeight:
                    return True

            # After checking all workers, return True if the total reduction is sufficient.
            return total_reduction >= mountainHeight

        # Perform binary search for the minimum time T in the range [0, T_upper].
        low = 0
        high = T_upper
        min_T = high # Initialize the minimum time found to the upper bound.

        while low <= high:
            # Calculate midpoint safely to avoid potential overflow if low+high is large
            mid = low + (high - low) // 2
            if check(mid):
                # If time 'mid' is sufficient, it becomes a candidate for the minimum time.
                # Store it and try searching in the lower half [low, mid-1].
                min_T = mid
                high = mid - 1
            else:
                # If time 'mid' is not sufficient, we need more time.
                # Search in the upper half [mid+1, high].
                low = mid + 1
                
        # The loop terminates when low > high. 'min_T' holds the minimum sufficient time found.
        return min_T