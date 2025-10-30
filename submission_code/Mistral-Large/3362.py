from typing import List
import bisect

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        distinct_counts = []

        # Helper function to compute distinct counts for all subarrays
        def compute_distinct_counts():
            for start in range(n):
                count_map = {}
                current_count = 0
                for end in range(start, n):
                    if nums[end] not in count_map:
                        count_map[nums[end]] = 1
                        current_count += 1
                    else:
                        count_map[nums[end]] += 1
                    distinct_counts.append(current_count)

        # Compute all distinct counts
        compute_distinct_counts()

        # Sort the distinct counts array
        distinct_counts.sort()

        # Find the median
        median_index = len(distinct_counts) // 2
        return distinct_counts[median_index]

# Example usage:
# solution = Solution()
# print(solution.medianOfUniquenessArray([1, 2, 3]))  # Output: 1
# print(solution.medianOfUniquenessArray([3, 4, 3, 4, 5]))  # Output: 2
# print(solution.medianOfUniquenessArray([4, 3, 5, 4]))  # Output: 2