class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            consecutive = True
            for j in range(i, i + k - 1):
                if nums[j+1] - nums[j] != 1:
                    consecutive = False
                    break
            if consecutive:
                # In a consecutively ascending subarray, the maximum is simply the last element
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
                
        return results