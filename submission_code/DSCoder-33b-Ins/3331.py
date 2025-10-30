class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        operations = 0
        while nums[0] < k:
            if nums[1] < k:
                nums[1] = nums[0] + nums[1]
                nums.pop(0)
            else:
                nums[0] = nums[0] + nums[1]
                nums.pop(1)
            nums.sort()
            operations += 1
        return operations