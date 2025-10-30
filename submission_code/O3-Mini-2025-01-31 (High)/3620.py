from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Sort nums. For each number x we can transform it to any value in [x-k, x+k].
        # We want to choose a distinct number from each interval (if possible)
        # so that the total number of distinct outcomes is maximized.
        # This is equivalent to an interval scheduling (or matching) problem where
        # we assign an integer to each interval such that the assigned integers are strictly increasing.
        # The greedy strategy is to sort intervals by their right endpoint and then, for each interval,
        # choose the smallest number that is larger than the last chosen number and within the interval.
        
        nums.sort()  # sorting by x effectively sorts intervals [x-k, x+k] by their right endpoint (x+k)
        count = 0
        # Initialize last assigned number to a very low integer.
        last_assigned = -10**18
        
        for x in nums:
            L, R = x - k, x + k
            # The candidate is the smallest number we can pick that is > last_assigned and within [L, R].
            candidate = max(L, last_assigned + 1)
            if candidate <= R:
                count += 1
                last_assigned = candidate
        return count

# Example test cases
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxDistinctElements([1, 2, 2, 3, 3, 4], 2))  # Expected output: 6
    print(sol.maxDistinctElements([4, 4, 4, 4], 1))           # Expected output: 3