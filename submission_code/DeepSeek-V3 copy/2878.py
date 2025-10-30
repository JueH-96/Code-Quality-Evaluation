class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        current = 0
        
        for i in range(n):
            current += diff[i]
            nums[i] -= current
            if nums[i] < 0:
                return False
            if nums[i] > 0:
                if i + k > n:
                    return False
                current += nums[i]
                if i + k < n:
                    diff[i + k] -= nums[i]
                nums[i] = 0
        
        return True