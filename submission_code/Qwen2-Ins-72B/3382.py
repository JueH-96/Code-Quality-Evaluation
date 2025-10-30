class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        max_num = max(nums)
        left = 0
        right = 0
        count = 0
        subarray_count = 0
        while right < n:
            if nums[right] == max_num:
                count = 0
            else:
                count += 1
            while nums[left] != max_num or count >= right - left:
                if nums[left] == max_num:
                    count -= 1
                left += 1
            subarray_count += left
            right += 1
        return subarray_count