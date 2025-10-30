class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(n):
            return max(str(n))

        max_sums = {}
        for num in nums:
            max_d = max_digit(num)
            if max_d not in max_sums:
                max_sums[max_d] = []
            max_sums[max_d].append(num)

        ans = -1
        for max_d in max_sums:
            if len(max_sums[max_d]) >= 2:
                max_sums[max_d].sort()
                ans = max(ans, max_sums[max_d][-1] + max_sums[max_d][-2])

        return ans