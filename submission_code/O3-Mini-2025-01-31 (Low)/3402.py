from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        # Since we can only increase numbers,
        # the natural target is max(nums).
        target = max(nums)
        
        # Calculate deficits for each element.
        diffs = [target - num for num in nums]
        total_diff = sum(diffs)
        # If no deficit then cost is 0.
        if total_diff == 0:
            return 0
        
        # When doing two single operations separately the cost would be 2*cost1.
        # So if cost2 is not beneficial, we simply do singles.
        if cost2 >= 2 * cost1:
            return (total_diff * cost1) % MOD
        
        # Otherwise, we want to use as many double-increment operations as possible.
        # However, we can only combine increments that occur at two different indices.
        # A useful observation: if one index needs a lot of increments and
        # every other index needs only a few, then that index can only be paired as many times
        # as the sum of the deficits of the other indices.
        # So the maximum number of double operations is limited by:
        #    min(floor(total_diff/2), total_diff - max(diffs))
        max_diff = max(diffs)
        max_pairs = min(total_diff // 2, total_diff - max_diff)
        
        # Use the maximum possible pairing
        pairs_used = max_pairs
        singles_needed = total_diff - 2 * pairs_used
        cost = (pairs_used * cost2 + singles_needed * cost1) % MOD
        return cost

# For testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minCostToEqualizeArray([4,1], 5, 2))  # Expected Output: 15
    # Example 2:
    print(sol.minCostToEqualizeArray([2,3,3,3,5], 2, 1))  # Expected Output: 6
    # Example 3:
    print(sol.minCostToEqualizeArray([3,5,3], 1, 3))  # Expected Output: 4