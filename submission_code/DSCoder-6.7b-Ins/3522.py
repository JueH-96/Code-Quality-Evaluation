class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = [0] * (n - k + 1)

        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if subarray == sorted(subarray) and subarray == [min(subarray), max(subarray)]:
                results[i] = max(subarray)
            else:
                results[i] = -1

        return results