class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        distinct_count = 1

        for i in range(1, n):
            if nums[i] != nums[i - 1]:
                distinct_count += 1

        if k == 0:
            return distinct_count

        operations = 0
        i = 1
        while i < n and operations < k:
            if nums[i] == nums[i - 1]:
                diff = nums[i] - nums[i - 1]
                if diff <= k - operations:
                    operations += diff
                    distinct_count -= 1
                else:
                    break
            i += 1

        return distinct_count