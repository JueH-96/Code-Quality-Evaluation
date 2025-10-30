class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7
        total_power_sum = 0

        def find_subsequences_with_sum(index, current_sum, current_subsequence):
            nonlocal total_power_sum
            if current_sum == k:
                m = len(current_subsequence)
                contribution = pow(2, n - m, mod)
                total_power_sum = (total_power_sum + contribution) % mod
                return

            if index == n or current_sum > k:
                return

            # Include nums[index]
            find_subsequences_with_sum(index + 1, current_sum + nums[index], current_subsequence + [nums[index]])

            # Exclude nums[index]
            find_subsequences_with_sum(index + 1, current_sum, current_subsequence)

        find_subsequences_with_sum(0, 0, [])
        return total_power_sum