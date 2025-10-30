import collections
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of groups for a valid assignment.

        The solution strategy is as follows:
        1. Count the frequencies of each number in `nums`. Let these counts be c_1, c_2, ...
        2. A valid assignment requires all groups to have sizes k or k+1 for some integer k.
        3. For a given k, a count 'c' can be partitioned into groups of size k or k+1 if and only if c // k >= c % k.
        4. The goal is to find the largest k that is valid for all counts, as this minimizes the total number of groups.
        5. The search space for k is from min(counts) down to 1.
        6. Once the optimal k is found, the total number of groups is the sum of ceil(c / (k + 1)) for each count c.
        """

        # Step 1: Count the frequencies of each number.
        counts_map = collections.Counter(nums)
        counts = list(counts_map.values())

        # Step 2: The maximum possible value for k is the minimum count.
        # If k were larger, the smallest count could not be partitioned.
        c_min = min(counts)

        # Step 3: Iterate k downwards from c_min to 1 to find the largest valid k.
        for k in range(c_min, 0, -1):
            is_k_valid_for_all = True
            
            # Step 4: Check if this k is a valid basis for all counts.
            for c in counts:
                # The partition is possible if and only if: c // k >= c % k
                if c // k < c % k:
                    is_k_valid_for_all = False
                    break
            
            # Step 5: If k is valid, it's the optimal k. Calculate total groups and return.
            if is_k_valid_for_all:
                total_groups = 0
                for c in counts:
                    # Number of groups for count c is ceil(c / (k + 1)).
                    # Using integer arithmetic: (c + (k+1) - 1) // (k+1)
                    total_groups += (c + k) // (k + 1)
                return total_groups
        
        # This part is unreachable as k=1 is always a valid basis,
        # guaranteeing a solution is found in the loop.
        return -1