class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0

        for i in range(n - m):
            match = True
            for k in range(m):
                diff = nums[i + k + 1] - nums[i + k]
                if pattern[k] == 1:
                    if diff <= 0:
                        match = False
                        break
                elif pattern[k] == 0:
                    if diff != 0:
                        match = False
                        break
                elif pattern[k] == -1:
                    if diff >= 0:
                        match = False
                        break
            if match:
                count += 1

        return count