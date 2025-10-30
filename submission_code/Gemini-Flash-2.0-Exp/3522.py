class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            
            is_consecutive = True
            is_sorted = True
            
            for j in range(1, k):
                if subarray[j] != subarray[j-1] + 1:
                    is_consecutive = False
                    break
                if subarray[j] < subarray[j-1]:
                    is_sorted = False
                    break
            
            if is_consecutive and is_sorted:
                results.append(max(subarray))
            else:
                results.append(-1)
        
        return results