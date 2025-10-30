class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        if not nums:
            return 0
        add = nums[0]
        subtract = float('-inf')
        max_prev = add
        for num in nums[1:]:
            new_add = max(subtract + num, max_prev + num)
            new_subtract = add - num
            new_max_prev = max(new_add, new_subtract)
            add, subtract, max_prev = new_add, new_subtract, new_max_prev
        return max_prev