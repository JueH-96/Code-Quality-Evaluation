class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        num_set = set()
        for num in nums:
            if num in num_set:
                return False
            else:
                num_set.add(num)
        return len(num_set) >= len(nums) / 2