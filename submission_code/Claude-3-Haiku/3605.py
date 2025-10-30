class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1] * n
        
        for i in range(n):
            for j in range(nums[i]):
                if (j) | (j + 1) == nums[i]:
                    ans[i] = j
                    break
        
        return ans