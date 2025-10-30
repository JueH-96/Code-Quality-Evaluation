from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # We are allowed to delete any elements and then choose a contiguous segment from
        # the remaining array such that all its elements are unique.
        # Because deletion is free (other than not deleting all elements), we can choose
        # any subset of elements (while preserving order) and then “compress” them into a contiguous block.
        # In other words, we only want to choose at most one copy of each number.
        #
        # In our task we want to maximize the sum.
        # The best strategy is:
        #   1. For every distinct element in nums, if it makes a positive contribution (i.e. > 0), we want to include it.
        #   2. Zero does not change the sum so including it does no harm, but note that adding it does not improve
        #      the sum either.
        #   3. We are not forced to include any negative numbers since they would lower the sum.
        #
        # However, if the array contains no positive numbers (and possibly just negatives or just 0),
        # we must choose at least one element (cannot be empty). In this case, the best result will be the maximum element.
        
        distinct = set(nums)
        positive_sum = sum(x for x in distinct if x > 0)
        
        if positive_sum > 0:
            return positive_sum
        else:
            # If there is no positive number, the best we can do is select the maximum (which could be 0 or a negative)
            return max(distinct)

# Below are some simple tests

if __name__ == '__main__':
    sol = Solution()
    # Example 1
    print(sol.maxSum([1, 2, 3, 4, 5]))  # Expected output: 15
    # Example 2
    print(sol.maxSum([1, 1, 0, 1, 1]))  # Expected output: 1
    # Example 3
    print(sol.maxSum([1, 2, -1, -2, 1, 0, -1]))  # Expected output: 3
    # Additional tests
    print(sol.maxSum([-1, -2, -3]))  # Expected output: -1 (best is to choose the maximum element among negatives)
    print(sol.maxSum([-1, 0, -2]))   # Expected output: 0 (0 is better than negatives)