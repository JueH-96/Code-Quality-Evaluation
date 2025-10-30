class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        def is_consecutive(arr):
            return sorted(arr) == list(range(min(arr), max(arr) + 1))
        
        results = []
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i + k]
            if is_consecutive(subarray):
                results.append(max(subarray))
            else:
                results.append(-1)
        return results