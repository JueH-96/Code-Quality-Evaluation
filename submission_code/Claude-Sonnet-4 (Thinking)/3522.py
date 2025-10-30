class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            # Check if subarray nums[i:i+k] is consecutive and sorted
            is_valid = True
            for j in range(i + 1, i + k):
                if nums[j] != nums[j-1] + 1:
                    is_valid = False
                    break
            
            if is_valid:
                results.append(nums[i + k - 1])  # last element is the maximum
            else:
                results.append(-1)
        
        return results