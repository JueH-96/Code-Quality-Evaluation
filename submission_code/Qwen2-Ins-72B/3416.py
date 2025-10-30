class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(len(str(nums[0]))):
            cnt = [0] * 10
            for num in nums:
                cnt[int(str(num)[i])] += 1
            for j in range(10):
                ans += j * cnt[j] * (n - cnt[j])
        return ans