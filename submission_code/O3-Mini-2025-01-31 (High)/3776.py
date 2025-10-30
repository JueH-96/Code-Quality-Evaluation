from functools import lru_cache
from typing import List

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # dp(i, front) returns the minimum cost to remove all elements
        # if we have already processed nums[0:i] and 'front' (a tuple) is what remains in front.
        @lru_cache(maxsize=None)
        def dp(i: int, front: tuple) -> int:
            # First, "fill" the front until we have 3 elements (if possible).
            current = list(front)
            j = i
            while len(current) < 3 and j < n:
                current.append(nums[j])
                j += 1
            current = tuple(current)
            
            # If fewer than 3 elements remain, we must remove them all in one operation.
            if len(current) < 3:
                return max(current) if current else 0
            
            # Now, we have exactly 3 elements, say a, b, c.
            a, b, c = current
            
            # We have three choices:
            # 1. Remove a and b; cost is max(a, b), and new front becomes (c,)
            cost1 = max(a, b) + dp(j, (c,))
            
            # 2. Remove a and c; cost is max(a, c), and new front becomes (b,)
            cost2 = max(a, c) + dp(j, (b,))
            
            # 3. Remove b and c; cost is max(b, c), and new front becomes (a,)
            cost3 = max(b, c) + dp(j, (a,))
            
            return min(cost1, cost2, cost3)
        
        # Initially, no extra elements in front and we start at index 0.
        return dp(0, ())

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minCost([6, 2, 8, 4]))  # Expected output: 12
    # Example 2
    print(sol.minCost([2, 1, 3, 3]))  # Expected output: 5