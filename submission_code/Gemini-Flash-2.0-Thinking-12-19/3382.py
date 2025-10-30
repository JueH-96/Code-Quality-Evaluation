class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            max_so_far = 0
            for j in range(i, n):
                max_so_far = max(max_so_far, nums[j])
                if nums[i] == max_so_far and nums[j] == max_so_far:
                    count += 1
        return count