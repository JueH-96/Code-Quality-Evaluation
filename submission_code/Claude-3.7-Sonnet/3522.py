class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            
            # Check if the subarray is sorted in ascending order
            is_sorted = all(subarray[j] < subarray[j+1] for j in range(k-1))
            
            if is_sorted:
                # Check if elements are consecutive
                is_consecutive = all(subarray[j+1] - subarray[j] == 1 for j in range(k-1))
                
                if is_consecutive:
                    results.append(max(subarray))
                else:
                    results.append(-1)
            else:
                results.append(-1)
        
        return results