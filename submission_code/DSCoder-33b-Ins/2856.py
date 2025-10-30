class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        distinct_elements = len(set(nums))
        count = 0
        for i in range(n):
            distinct_set = set()
            for j in range(i, n):
                distinct_set.add(nums[j])
                if len(distinct_set) == distinct_elements:
                    count += 1
        return count