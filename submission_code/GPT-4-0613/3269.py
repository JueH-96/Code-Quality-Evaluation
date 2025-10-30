class Solution:
    def countMatchingSubarrays(self, nums: list[int], pattern: list[int]) -> int:
        count = 0
        for i in range(len(nums) - len(pattern)):
            match = True
            for j in range(len(pattern)):
                if pattern[j] == 1 and nums[i + j + 1] <= nums[i + j]:
                    match = False
                    break
                elif pattern[j] == 0 and nums[i + j + 1] != nums[i + j]:
                    match = False
                    break
                elif pattern[j] == -1 and nums[i + j + 1] >= nums[i + j]:
                    match = False
                    break
            if match:
                count += 1
        return count