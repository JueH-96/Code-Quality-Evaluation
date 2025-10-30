class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def matches_pattern(subarray, pattern):
            for i in range(len(pattern)):
                if pattern[i] == 1 and subarray[i + 1] <= subarray[i]:
                    return False
                if pattern[i] == 0 and subarray[i + 1] != subarray[i]:
                    return False
                if pattern[i] == -1 and subarray[i + 1] >= subarray[i]:
                    return False
            return True

        n = len(nums)
        m = len(pattern)
        count = 0

        for i in range(n - m):
            subarray = nums[i:i + m + 1]
            if matches_pattern(subarray, pattern):
                count += 1

        return count