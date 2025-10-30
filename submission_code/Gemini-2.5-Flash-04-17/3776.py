from typing import List
from functools import lru_cache

class Solution:
    def minCost(self, nums: List[int]) -> int:
        N = len(nums)

        # Base case: If fewer than 3 elements initially, remove all.
        # The cost is the maximum of the remaining elements.
        # Since 1 <= nums.length <= 1000, nums is never empty.
        if N <= 2:
            return max(nums)

        # Memoization decorator for the recursive function
        # State: solve(i, j) represents the minimum cost to remove the remaining elements,
        # where the current effective array consists of nums[j] followed by nums[i:]
        # (original indices).
        # i: original index of the start of the contiguous block nums[i:]
        #    (i >= 3 in this DP as the first step is handled outside)
        # j: original index of the element kept from the previous step (j < i, j >= 0)
        #    This element nums[j] is the first element of the current effective array.

        @lru_cache(None)
        def solve(i: int, j: int) -> int:
            # Number of elements in the contiguous block = N - i
            # Current effective array elements are [nums[j]] followed by nums[i:]
            # Total elements remaining = 1 (from j) + (N - i)
            rem_len = 1 + N - i

            # Base Case: If fewer than 3 elements remain in the current effective array
            # This condition is 1 + (N - i) <= 2, which simplifies to N - i <= 1.
            if N - i <= 1:
                # Remaining elements are [nums[j]] + nums[i:N]
                current_elements_values = [nums[j]] + nums[i:]
                # current_elements_values will not be empty here since N >= 1 and j >= 0.
                # The smallest rem_len handled by this base case is 1 (when N-i=0).
                return max(current_elements_values)

            # Recursive Step: 3 or more elements remain in the current effective array
            # rem_len = 1 + N - i >= 3 => N - i >= 2
            # This guarantees i <= N - 2, so i and i+1 are valid indices in nums.
            # The first three elements of the current effective array are nums[j], nums[i], nums[i+1]
            # Their original indices are j, i, i+1
            o0, o1, o2 = j, i, i + 1

            # After removing two elements from these three, the next two elements will come from the contiguous block starting after nums[i+1].
            # This means the new contiguous block starts at original index i+2.
            next_i = i + 2

            # We have 3 options for removing two elements from the first three (nums[o0], nums[o1], nums[o2]):

            # Option 1: Remove (o0, o1), keep o2. Cost is max(nums[o0], nums[o1]).
            # The remaining array starts with nums[o2] followed by nums[next_i:]
            # The next state is solve(next_i, o2).
            cost1 = max(nums[o0], nums[o1]) + solve(next_i, o2)

            # Option 2: Remove (o0, o2), keep o1. Cost is max(nums[o0], nums[o2]).
            # The remaining array starts with nums[o1] followed by nums[next_i:]
            # The next state is solve(next_i, o1).
            cost2 = max(nums[o0], nums[o2]) + solve(next_i, o1)

            # Option 3: Remove (o1, o2), keep o0. Cost is max(nums[o1], nums[o2]).
            # The remaining array starts with nums[o0] followed by nums[next_i:]
            # The next state is solve(next_i, o0).
            cost3 = max(nums[o1], nums[o2]) + solve(next_i, o0)

            # Return the minimum cost among the three options
            return min(cost1, cost2, cost3)

        # Handle the very first step explicitly when N >= 3
        # The initial array is nums[0], nums[1], ..., nums[N-1].
        # The first three elements are nums[0], nums[1], nums[2].
        # We must choose two of these to remove.

        # After the first operation, the array will consist of one element (kept from original 0,1,2)
        # followed by original elements starting from index 3. This structure matches the DP state solve(i, j)
        # where i=3 and j is the original index of the kept element (0, 1, or 2).

        # Option A: Remove nums[0] and nums[1]. Cost is max(nums[0], nums[1]).
        # nums[2] (original index 2) is kept. The remaining original elements start from index 3.
        # The next state is solve(3, 2).
        cost_A = max(nums[0], nums[1]) + solve(3, 2)

        # Option B: Remove nums[0] and nums[2]. Cost is max(nums[0], nums[2]).
        # nums[1] (original index 1) is kept. Remaining original elements start from index 3.
        # The next state is solve(3, 1).
        cost_B = max(nums[0], nums[2]) + solve(3, 1)

        # Option C: Remove nums[1] and nums[2]. Cost is max(nums[1], nums[2]).
        # nums[0] (original index 0) is kept. Remaining original elements start from index 3.
        # The next state is solve(3, 0).
        cost_C = max(nums[1], nums[2]) + solve(3, 0)

        # The minimum cost is the minimum of the costs from these three initial options.
        return min(cost_A, cost_B, cost_C)