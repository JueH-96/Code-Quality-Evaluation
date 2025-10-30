import sys
from functools import cache
from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        infinity = float('inf')

        # @cache provides memoization for the recursive function.
        # The state is (i, j), representing the subproblem of partitioning
        # nums[i:] to satisfy the requirements of andValues[j:].
        @cache
        def solve(i: int, j: int) -> float:
            """
            Calculates the minimum value sum for partitioning the subarray
            nums[i:] to satisfy the andValues[j:].
            """
            # Base case: If we have successfully formed all m subarrays.
            if j == m:
                # A valid partition is found only if we have also used up all elements of nums.
                return 0 if i == n else infinity

            # Base case: If we have run out of numbers but still need to form subarrays.
            if i == n:
                return infinity

            min_sum_for_this_state = infinity
            # -1 in two's complement has all bits set to 1, which is the identity for bitwise AND.
            current_and = -1

            # The j-th subarray ends at index k. The remaining m-(j+1) subarrays must
            # fit in the remaining n-(k+1) elements. So, n-k-1 >= m-j-1, which means k <= n-(m-j).
            for k in range(i, n - (m - j) + 1):
                current_and &= nums[k]

                # Optimization: The running AND is non-increasing. If it ever lacks a bit
                # that andValues[j] requires, it can never satisfy the condition later.
                # This is because ANDing with more numbers can only turn bits from 1 to 0.
                if (current_and & andValues[j]) != andValues[j]:
                    break
                
                # If the AND of the current subarray matches the target andValue.
                if current_and == andValues[j]:
                    # Recursively solve for the rest of the arrays.
                    sub_problem_result = solve(k + 1, j + 1)
                    
                    # If the subproblem has a valid solution.
                    if sub_problem_result != infinity:
                        # The value of the current subarray is its last element, nums[k].
                        current_total_sum = nums[k] + sub_problem_result
                        min_sum_for_this_state = min(min_sum_for_this_state, current_total_sum)
            
            return min_sum_for_this_state

        # Start the recursion from the beginning of both arrays.
        result = solve(0, 0)
        
        # If the result is infinity, no valid partition was found.
        return int(result) if result != infinity else -1