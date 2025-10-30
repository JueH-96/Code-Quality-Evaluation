class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while any(nums):
            max_nums = []
            for row in nums:
                if row:
                    max_nums.append(max(row))
                else:
                    max_nums.append(0)

            max_val = max(max_nums)
            score += max_val

            for i in range(len(nums)):
                if nums[i]:
                    if max(nums[i]) == max_val:
                        nums[i].remove(max(nums[i]))
                    

        return score