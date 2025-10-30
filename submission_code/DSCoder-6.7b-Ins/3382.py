class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        max_num = max(nums)
        count = [0] * (n + 1)
        pos = [0] * (n + 1)
        count[0] = 1
        pos[0] = -1
        for i in range(n):
            if nums[i] == max_num:
                pos[i + 1] = i + 1
            else:
                pos[i + 1] = pos[i]
            count[i + 1] = count[i] + (i - pos[i] + 1)
        return count[-1]