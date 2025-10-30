class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def greatest_proper_divisor(x):
            if x == 1:
                return 1
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0:
                    return x // i
            return 1

        n = len(nums)
        operations = 0

        for i in range(n - 1):
            while nums[i] > nums[i + 1]:
                gpd = greatest_proper_divisor(nums[i])
                if gpd == nums[i]:
                    return -1
                nums[i] = gpd
                operations += 1

        return operations

# Example usage:
# sol = Solution()
# print(sol.minOperations([25, 7]))  # Output: 1
# print(sol.minOperations([7, 7, 6]))  # Output: -1
# print(sol.minOperations([1, 1, 1, 1]))  # Output: 0