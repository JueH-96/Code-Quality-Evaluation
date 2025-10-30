class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                if not sub_array:
                    continue
                max_val = max(sub_array)
                if sub_array[0] == max_val and sub_array[-1] == max_val:
                    count += 1
        return count