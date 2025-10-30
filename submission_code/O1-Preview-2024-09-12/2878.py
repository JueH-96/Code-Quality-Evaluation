class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0]*(n+1)
        res = 0
        for i in range(n):
            res += diff[i]
            need = nums[i] + res
            if need < 0:
                return False
            if need == 0:
                continue
            if i + k > n:
                return False
            res -= need
            diff[i + k] += need
        return True