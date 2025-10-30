class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_val = max(sub_array)
                if sub_array[0] == max_val and sub_array[-1] == max_val:
                    ans += 1
        return ans