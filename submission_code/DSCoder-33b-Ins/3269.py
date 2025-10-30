class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        for i in range(len(nums) - len(pattern) + 1):
            match = True
            for j in range(len(pattern) - 1):
                if pattern[j] == 1 and nums[i + j + 1] <= nums[i + j]:
                    match = False
                    break
                elif pattern[j] == -1 and nums[i + j + 1] >= nums[i + j]:
                    match = False
                    break
                elif pattern[j] == 0 and nums[i + j + 1] != nums[i + j]:
                    match = False
                    break
            if match:
                count += 1
        return count