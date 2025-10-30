class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def get_pattern(arr):
            return [1 if arr[i] > arr[i + 1] else 0 if arr[i] == arr[i + 1] else -1 for i in range(len(arr) - 1)]

        n = len(nums)
        m = len(pattern)
        count = 0

        for i in range(n - m):
            if get_pattern(nums[i:i + m + 1]) == pattern:
                count += 1

        return count