class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_count = 0
                for num in sub_array:
                    if num == max_val:
                        max_count += 1
                if max_count >= k:
                    count += 1
        return count