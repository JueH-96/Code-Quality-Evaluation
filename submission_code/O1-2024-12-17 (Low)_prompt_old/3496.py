class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        import math
        
        # We want to find the smallest T such that
        # sum of the maximum height each worker can reduce at time T
        # (which we'll call hi for worker i) is >= mountainHeight
        #
        # For worker i with time cost w, if w * (1 + 2 + ... + hi) <= T,
        # that means w * hi*(hi+1)/2 <= T.
        # So hi*(hi+1) <= 2T / w.
        #
        # Solve hi^2 + hi - (2T / w) <= 0 for hi, we get hi <= floor((-1 + sqrt(1 + 8T/w))/2).
        
        def canReduceToZero(T: int) -> bool:
            # Compute how many units of height can be removed in time T
            total_removed = 0
            for w in workerTimes:
                # Solve for the maximum hi such that hi*(hi+1)/2 * w <= T
                # => hi*(hi+1) <= 2T / w
                if w == 0:  # (Though by constraints, w >= 1, this is just a safe check)
                    continue
                # timeBound = (2T / w)
                # hi = floor((-1 + sqrt(1 + 8 * T / w)) / 2)
                max_hi = int((-1 + math.isqrt(1 + 8 * (T // w))) // 2) if (T // w) > 0 else 0
                
                # Because float-based or direct T/w might be large, let's be consistent:
                # We can also do float but handle large T carefully using math.isqrt.
                # A small correction when using T//w can lead to an off-by-one, so let's adjust carefully:
                # We'll do a small while loop to correct off-by-one if needed.
                
                # Correct potential off-by-one from integer division approach:
                while max_hi * (max_hi + 1) // 2 * w > T and max_hi > 0:
                    max_hi -= 1
                while (max_hi + 1) * (max_hi + 2) // 2 * w <= T:
                    max_hi += 1
                
                total_removed += max_hi
                if total_removed >= mountainHeight:
                    return True
            return total_removed >= mountainHeight

        # The maximum time if only the slowest worker does all the work:
        # worst_possible_time = max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        # But to be safe, we can assign an upper bound:
        max_w = max(workerTimes)
        worst_possible_time = max_w * (mountainHeight * (mountainHeight + 1) // 2)
        
        # Binary search in [0, worst_possible_time]
        left, right = 0, worst_possible_time
        answer = worst_possible_time
        
        while left <= right:
            mid = (left + right) // 2
            if canReduceToZero(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer