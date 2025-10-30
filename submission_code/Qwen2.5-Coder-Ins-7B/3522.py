class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = [-1] * (n - k + 1)
        
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if self.isConsecutiveAndSorted(subarray):
                results[i] = max(subarray)
        
        return results
    
    def isConsecutiveAndSorted(self, arr: List[int]) -> bool:
        return arr == sorted(arr) and len(set(arr)) == len(arr)