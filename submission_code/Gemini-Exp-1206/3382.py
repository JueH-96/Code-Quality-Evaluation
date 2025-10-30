class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_val = 0
                for num in sub_array:
                    max_val = max(max_val, num)
                
                if sub_array[0] == max_val and sub_array[-1] == max_val:
                    count += 1
        return count