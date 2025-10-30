class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        def get_sum(l, r):
            return prefix_sum[r + 1] - prefix_sum[l]

        def get_cost(l, r):
            return get_sum(l, r) * ((r - l) % 2 * 2 - 1)

        left, right = 0, n - 1
        total_cost = 0
        while left <= right:
            if get_cost(0, left) + get_cost(right, n - 1) > get_cost(0, right) + get_cost(left, n - 1):
                total_cost += get_sum(left, right)
                right -= 1
            else:
                total_cost += get_sum(left, right)
                left += 1
        return total_cost