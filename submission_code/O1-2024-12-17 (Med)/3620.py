class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Each element nums[i] can become any integer in [nums[i] - k, nums[i] + k].
        # We want to assign a distinct integer to as many elements as possible.
        # This is equivalent to the problem:
        # "Given intervals [l_i, r_i], assign a unique integer within each interval
        #  to maximize the number of assigned intervals."
        #
        # A known greedy strategy (sometimes called "interval scheduling to distinct points")
        # is to sort intervals by their right endpoint and then always pick the smallest integer
        # that is >= the last chosen integer + 1 and also >= the current interval's left boundary.
        
        # Build intervals [x-k, x+k] for each x in nums.
        intervals = [(x - k, x + k) for x in nums]
        
        # Sort intervals by their right endpoint.
        intervals.sort(key=lambda interval: interval[1])
        
        count = 0
        # We'll keep track of the last assigned integer in "chosen".
        # Pick a very small number initially so that we can pick something
        # as soon as possible.
        chosen = -10**20
        
        # Iterate over intervals in ascending order of their right endpoint.
        for left, right in intervals:
            # The next assignable integer should be at least (chosen + 1).
            # Also, it must be at least "left" to stay within the interval.
            candidate = max(chosen + 1, left)
            
            # If that candidate is within the current interval, assign it.
            if candidate <= right:
                chosen = candidate
                count += 1
        
        return count