class Solution:
    def resultsArray(self, nums: list[int], k: int) -> list[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if max(subarray) - min(subarray) == k - 1 and len(set(subarray)) == k:
                results.append(max(subarray))
            else:
                results.append(-1)
        return results