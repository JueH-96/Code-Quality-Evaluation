class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_element = max(sub_array)
                if sub_array[0] == max_element and sub_array[-1] == max_element:
                    count += 1
        return count