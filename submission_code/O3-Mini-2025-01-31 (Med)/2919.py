from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        
        # Helper function to check if we can form m groups.
        # For m groups, the sizes must be 1, 2, …, m, so total required elements are m*(m+1)//2.
        # However, note that each number i can be used in at most one position per group.
        # That means, even if usageLimits[i] is huge, it can only contribute to at most m groups.
        # So the effective capacity from a number is: min(usageLimits[i], m)
        # Summing up over all numbers, if sum(min(usageLimits[i], m)) is >= m*(m+1)//2, then forming m groups is feasible.
        def feasible(m: int) -> bool:
            required = m * (m + 1) // 2
            total = 0
            for limit in usageLimits:
                # Each number can contribute at most once per group.
                total += min(limit, m)
                if total >= required:  # Early exit if we've already covered the required count.
                    return True
            return total >= required
        
        # Binary search for the maximum m (number of groups) that is feasible.
        # Note that in the ideal case where each number has huge limits, a number i can be used in at most m groups.
        # With n numbers, the maximum total contribution is n * m.
        # We need m*(m+1)//2 <= n*m  => m+1 <= 2*n, so m is at most about 2*n.
        low, high = 0, 2 * n + 1
        
        while low < high:
            mid = (low + high) // 2
            if feasible(mid):
                low = mid + 1  # Try for a larger m if mid groups is feasible.
            else:
                high = mid   # Otherwise reduce the search space.
                
        return low - 1  # low is one greater than the maximum feasible m.

# Test cases to validate the solution
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.maxIncreasingGroups([1,2,5]))  # Expected output: 3
    # Example 2
    print(sol.maxIncreasingGroups([2,1,2]))  # Expected output: 2
    # Example 3
    print(sol.maxIncreasingGroups([1,1]))    # Expected output: 1