class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        drops = 0
        rotation_idx = -1
        
        for i in range(n):
            next_i = (i + 1) % n
            if nums[i] > nums[next_i]:
                drops += 1
                rotation_idx = i
                if drops > 1:
                    break
        
        if drops > 1:
            return -1
        elif drops == 0:
            return 0
        else:
            return (n - (rotation_idx + 1))