class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        
        for i in range(n - k + 1):
            # Get current window
            window = nums[i:i+k]
            
            # Check if sorted and consecutive
            is_valid = True
            for j in range(1, k):
                if window[j] != window[j-1] + 1:
                    is_valid = False
                    break
            
            # If valid, add max element, else add -1
            if is_valid:
                result.append(max(window))
            else:
                result.append(-1)
                
        return result