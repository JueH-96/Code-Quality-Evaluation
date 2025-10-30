class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        rotation_index = -1
        for i in range(n):
            if nums[i] > nums[(i+1)%n]:
                count +=1
                rotation_index = i
        if count == 0:
            return 0
        elif count == 1:
            return (n - (rotation_index + 1)) % n
        else:
            return -1