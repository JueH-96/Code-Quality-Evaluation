from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # There are at most n distinct numbers available.
        n = len(usageLimits)
        
        # Helper: check if it's possible to form `g` groups.
        # In g groups the groups sizes must be 1, 2, ..., g.
        # Total number of picks needed is: total_needed = g * (g+1) // 2.
        # Also, note that a number with limit x can contribute at most min(x, g)
        # (since even if x > g, it can appear in at most one pick per group).
        def feasible(g: int) -> bool:
            total_needed = g * (g + 1) // 2
            total_supply = 0
            for x in usageLimits:
                # Each number can contribute at most g times (one per group).
                total_supply += x if x < g else g
                # Early stop if we already have enough supply.
                if total_supply >= total_needed:
                    break
            return total_supply >= total_needed
        
        # Maximum possible groups cannot exceed n
        low, high = 0, n
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if feasible(mid):
                ans = mid   # mid groups can be formed; try for more
                low = mid + 1
            else:
                high = mid - 1
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Test Example 1:
    usageLimits1 = [1,2,5]
    print(sol.maxIncreasingGroups(usageLimits1))  # Expected output: 3

    # Test Example 2:
    usageLimits2 = [2,1,2]
    print(sol.maxIncreasingGroups(usageLimits2))  # Expected output: 2

    # Test Example 3:
    usageLimits3 = [1,1]
    print(sol.maxIncreasingGroups(usageLimits3))  # Expected output: 1