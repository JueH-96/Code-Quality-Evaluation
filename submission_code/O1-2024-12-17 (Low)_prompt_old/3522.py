class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for start in range(n - k + 1):
            # Check if subarray nums[start : start + k] is consecutive and ascending
            consecutive_ascending = True
            for i in range(start, start + k - 1):
                if nums[i + 1] != nums[i] + 1:
                    consecutive_ascending = False
                    break
            
            if consecutive_ascending:
                results.append(max(nums[start:start + k]))
            else:
                results.append(-1)
        
        return results