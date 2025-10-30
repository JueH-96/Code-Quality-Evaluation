from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # For each number, the possible range is [num - k, num + k].
        # The goal is to choose a distinct integer for as many numbers as possible,
        # so that each chosen integer belongs to the corresponding interval.
        # We can view it as a "greedy interval scheduling" problem: we want to assign
        # integers in increasing order such that the chosen integer for one interval
        # is strictly greater than the previously chosen one.
        #
        # How to solve:
        # 1. Build intervals: (left, right) for each number = (num - k, num + k).
        # 2. Sort intervals by their right endpoint (and then left for tie-breaking).
        # 3. Initialize "last" as something below possible minimum. For each interval,
        #    choose candidate = max(left, last+1). If candidate <= right then select candidate
        #    and update last = candidate.
        # 4. Count the number of intervals successfully assigned.
        #
        # This greedy strategy guarantees the maximum number of distinct assignments.
        
        # Build intervals: (left, right) for each number in nums.
        intervals = [(num - k, num + k) for num in nums]
        # Sort intervals by right endpoint, and then left endpoint.
        intervals.sort(key=lambda interval: (interval[1], interval[0]))
        
        last_assigned = -10**18  # Ensuring initial value is less than any possible candidate.
        count = 0
        
        for L, R in intervals:
            # The candidate must be strictly greater than last_assigned
            candidate = max(L, last_assigned + 1)
            if candidate <= R:
                count += 1
                last_assigned = candidate
        return count
                        
# Example usage:
# sol = Solution()
# print(sol.maxDistinctElements([1,2,2,3,3,4], 2))  # Output: 6
# print(sol.maxDistinctElements([4,4,4,4], 1))        # Output: 3