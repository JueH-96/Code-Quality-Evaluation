class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_val = 0
                for val in sub_array:
                    max_val = max(max_val, val)
                if sub_array[0] == max_val and sub_array[-1] == max_val:
                    count += 1
        return count