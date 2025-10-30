class Solution:
    def findKthSmallest(self, coins: list[int], k: int) -> int:
        # Helper function to count how many numbers are less than or equal to x
        def count_less_or_equal(x):
            count = 0
            for coin in coins:
                count += min(x // coin, k)  # We should not count more than k for any coin
            return count

        # Binary search to find the kth smallest number
        left, right = 1, coins[-1] * k
        while left < right:
            mid = (left + right) // 2
            if count_less_or_equal(mid) < k:
                left = mid + 1
            else:
                right = mid
        return left

# Example usage:
# sol = Solution()
# print(sol.findKthSmallest([3,6,9], 3))  # Output: 9
# print(sol.findKthSmallest([5,2], 7))    # Output: 12