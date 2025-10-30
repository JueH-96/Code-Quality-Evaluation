class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            # Check if the subarray from i to i+k-1 is consecutive and ascending
            is_valid = True
            
            for j in range(i + 1, i + k):
                if nums[j] != nums[j-1] + 1:
                    is_valid = False
                    break
            
            if is_valid:
                # The maximum element is the last element in the subarray
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        
        return results