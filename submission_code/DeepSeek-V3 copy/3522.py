class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            sorted_sub = sorted(subarray)
            if sorted_sub == list(range(sorted_sub[0], sorted_sub[0] + k)):
                results.append(sorted_sub[-1])
            else:
                results.append(-1)
        return results