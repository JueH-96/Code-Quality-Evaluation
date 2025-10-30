class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_distinct = len(set(nums))
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                if len(set(sub_array)) == total_distinct:
                    ans += 1
        return ans