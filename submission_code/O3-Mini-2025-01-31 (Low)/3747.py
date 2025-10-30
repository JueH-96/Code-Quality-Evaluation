class Solution:
    def maxAdjacentDistance(self, nums: list[int]) -> int:
        max_diff = 0
        n = len(nums)
        for i in range(n):
            diff = abs(nums[i] - nums[(i + 1) % n])
            if diff > max_diff:
                max_diff = diff
        return max_diff

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test case 1:
    print(sol.maxAdjacentDistance([1, 2, 4]))  # Expected output: 3
    # Test case 2:
    print(sol.maxAdjacentDistance([-5, -10, -5]))  # Expected output: 5