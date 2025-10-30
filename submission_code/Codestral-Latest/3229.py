class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]

        def cost_to_make_equal(nums, target):
            return sum(abs(num - target) for num in nums)

        min_cost = float('inf')
        for i in range(1, 10**9):
            if is_palindrome(i):
                min_cost = min(min_cost, cost_to_make_equal(nums, i))

        return min_cost