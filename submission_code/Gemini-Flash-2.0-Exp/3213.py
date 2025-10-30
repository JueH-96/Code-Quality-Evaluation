class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_num = max(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                max_count = 0
                for num in subarray:
                    if num == max_num:
                        max_count += 1
                if max_count >= k:
                    count += 1
        return count