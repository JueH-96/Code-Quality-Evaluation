class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        results = []
        n = len(nums)
        
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            max_elem = max(subarray)
            if sorted(subarray) == list(range(min(subarray), max_elem + 1)):
                results.append(max_elem)
            else:
                results.append(-1)
        
        return results