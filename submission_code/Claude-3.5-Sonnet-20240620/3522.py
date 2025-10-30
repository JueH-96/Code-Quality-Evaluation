class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            
            if all(subarray[j] == subarray[j-1] + 1 for j in range(1, k)):
                results.append(max(subarray))
            else:
                results.append(-1)
        
        return results