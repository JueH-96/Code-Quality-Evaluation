from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # We need to pick two breakpoints i and j such that
        # The three contiguous subarrays are:
        # nums[0:i], nums[i:j], nums[j:]
        # Their costs are: nums[0], nums[i], nums[j] respectively.
        # We need to minimize nums[0] + nums[i] + nums[j].
        # Since nums[0] is fixed, we only need to find the minimal possible nums[i] + nums[j]
        ans = float('inf')
        # i in [1, n - 2] and j in [i + 1, n - 1]
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cost = nums[0] + nums[i] + nums[j]
                ans = min(ans, cost)
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumCost([1, 2, 3, 12]))  # Output: 6
    print(sol.minimumCost([5, 4, 3]))        # Output: 12
    print(sol.minimumCost([10, 3, 1, 1]))    # Output: 12