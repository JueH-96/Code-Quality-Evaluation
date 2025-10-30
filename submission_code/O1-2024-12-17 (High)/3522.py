class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            consecutive = True
            for j in range(i + 1, i + k):
                if nums[j] != nums[j - 1] + 1:
                    consecutive = False
                    break
                    
            if consecutive:
                results.append(nums[i + k - 1])  # The maximum of a consecutive ascending subarray is its last element
            else:
                results.append(-1)
                
        return results