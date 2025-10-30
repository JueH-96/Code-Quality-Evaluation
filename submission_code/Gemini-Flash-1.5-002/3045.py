class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        shifts = 0
        
        if nums == sorted_nums:
            return 0

        for i in range(n):
            if nums == sorted_nums:
                return shifts
            
            temp = nums[-1]
            for j in range(n - 1, 0, -1):
                nums[j] = nums[j-1]
            nums[0] = temp
            shifts +=1
            if shifts > n:
                return -1

        if nums == sorted_nums:
            return shifts
        else:
            return -1