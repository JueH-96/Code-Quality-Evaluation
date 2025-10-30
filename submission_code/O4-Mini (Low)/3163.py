from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        # For each starting index i, expand the subarray to j
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                distinct_count = len(seen)
                total += distinct_count * distinct_count
        return total

# Example usage:
# sol = Solution()
# print(sol.sumCounts([1,2,1]))  # Output: 15
# print(sol.sumCounts([1,1]))    # Output: 3