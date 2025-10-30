class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        nums = list(nums) # Create a mutable copy
        for i in range(n):
            if nums[i] < 0:
                return False
            if nums[i] > 0:
                decrease_amount = nums[i]
                if i + k > n:
                    return False
                for j in range(k):
                    nums[i+j] -= decrease_amount
        for val in nums:
            if val != 0:
                return False
        return True