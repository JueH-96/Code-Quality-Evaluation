class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            if sorted(subarray) == list(range(subarray[0], subarray[-1]+1)):
                results.append(subarray[-1])
            else:
                results.append(-1)
        
        return results