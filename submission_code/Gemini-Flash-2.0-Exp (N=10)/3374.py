class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                is_alternating = True
                for k in range(i, j):
                    if nums[k] == nums[k+1]:
                        is_alternating = False
                        break
                if is_alternating:
                    count += 1
        return count