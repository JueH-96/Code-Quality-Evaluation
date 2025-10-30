class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        for i in range(1 << len(nums)):
            selected_nums = []
            for j in range(len(nums)):
                if (i >> j) & 1:
                    selected_nums.append(nums[j])
            if len(selected_nums) >= 2:
                bitwise_or = 0
                for num in selected_nums:
                    bitwise_or |= num
                if bitwise_or % 2 == 0 and bitwise_or >0:
                    return True
        return False