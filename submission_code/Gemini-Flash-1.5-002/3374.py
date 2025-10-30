class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sub_array = nums[i:j+1]
                is_alternating = True
                for k in range(len(sub_array)-1):
                    if sub_array[k] == sub_array[k+1]:
                        is_alternating = False
                        break
                if is_alternating:
                    count += 1
        return count