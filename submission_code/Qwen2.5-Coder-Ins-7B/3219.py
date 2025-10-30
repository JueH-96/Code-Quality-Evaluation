class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        indices = sorted(range(n), key=nums.__getitem__)
        result = [0] * n
        j = 0
        for i in range(n):
            if i == 0 or nums[indices[i]] - nums[indices[i-1]] > limit:
                j = i
            result[indices[j]] = nums[indices[i]]
        return result