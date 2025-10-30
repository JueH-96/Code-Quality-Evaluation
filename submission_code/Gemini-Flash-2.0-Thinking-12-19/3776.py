from typing import List
import functools

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)

        # Use tuple of original indices as the state for memoization
        # state: tuple[int, ...] representing the original indices of the remaining elements
        @functools.lru_cache(None)
        def solve(remaining_indices: tuple[int, ...]):
            k = len(remaining_indices) # number of remaining elements

            # Base cases: fewer than three elements remaining
            if k == 0:
                return 0
            if k == 1:
                # Cost is the value of the single remaining element
                return nums[remaining_indices[0]]
            if k == 2:
                # Cost is the maximum of the two remaining elements
                return max(nums[remaining_indices[0]], nums[remaining_indices[1]])

            # Recursive step for k >= 3: choose two elements from the first three
            # The first three *remaining* elements correspond to original indices remaining_indices[0], remaining_indices[1], remaining_indices[2]
            idx0 = remaining_indices[0]
            idx1 = remaining_indices[1]
            idx2 = remaining_indices[2]

            cost = float('inf')

            # Option 1: Remove the first two remaining elements (original indices idx0, idx1)
            # The new set of remaining indices starts from remaining_indices[2]
            # This corresponds to the tuple slicing remaining_indices[2:]
            new_indices_option1 = remaining_indices[2:]
            cost1 = max(nums[idx0], nums[idx1]) + solve(new_indices_option1)
            cost = min(cost, cost1)

            # Option 2: Remove the first and third remaining elements (original indices idx0, idx2)
            # The new set of remaining indices consists of remaining_indices[1] followed by remaining_indices[3:]
            new_indices_option2 = (idx1,) + remaining_indices[3:]
            cost2 = max(nums[idx0], nums[idx2]) + solve(new_indices_option2)
            cost = min(cost, cost2)

            # Option 3: Remove the second and third remaining elements (original indices idx1, idx2)
            # The new set of remaining indices consists of remaining_indices[0] followed by remaining_indices[3:]
            new_indices_option3 = (idx0,) + remaining_indices[3:]
            cost3 = max(nums[idx1], nums[idx2]) + solve(new_indices_option3)
            cost = min(cost, cost3)

            return cost

        # Initial state: tuple of all original indices from 0 to n-1
        initial_indices = tuple(range(n))
        return solve(initial_indices)