class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        mod = 10**9 + 7

        def count_subsequences(arr, target):
            dp = {}

            def solve(index, current_sum):
                if (index, current_sum) in dp:
                    return dp[(index, current_sum)]

                if current_sum == target:
                    return 1

                if index == len(arr):
                    return 0

                if current_sum > target:
                    return 0

                include = solve(index + 1, current_sum + arr[index])
                exclude = solve(index + 1, current_sum)

                dp[(index, current_sum)] = (include + exclude) % mod
                return dp[(index, current_sum)]

            return solve(0, 0)

        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])

            if subsequence:
                ans = (ans + count_subsequences(subsequence, k)) % mod

        return ans