class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        s1 = bin(nums[i])[2:]
                        s2 = bin(nums[j])[2:]
                        s3 = bin(nums[k])[2:]
                        ans = max(ans, int(s1 + s2 + s3, 2))
        return ans