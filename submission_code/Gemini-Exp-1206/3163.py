class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                distinct_count = len(set(sub_array))
                ans += distinct_count * distinct_count
        return ans