class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if subarray == sorted(subarray) and len(set(subarray)) == 1:
                results.append(subarray[0])
            else:
                results.append(-1)
        return results