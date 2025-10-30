class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                is_alternating = True
                if j > i:
                    for k in range(i + 1, j + 1):
                        if nums[k] == nums[k - 1]:
                            is_alternating = False
                            break
                if is_alternating:
                    count += 1
        return count