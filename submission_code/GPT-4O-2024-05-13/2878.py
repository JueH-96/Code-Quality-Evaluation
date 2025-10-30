class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        
        for i in range(n):
            if i > 0:
                diff[i] += diff[i - 1]
            nums[i] += diff[i]
            if nums[i] < 0:
                return False
            if nums[i] > 0:
                if i + k > n:
                    return False
                diff[i] -= nums[i]
                diff[i + k] += nums[i]
        
        return True