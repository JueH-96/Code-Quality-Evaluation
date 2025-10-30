class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            # Check if the subarray nums[i..(i+k-1)] is consecutive ascending
            is_consecutive = True
            for j in range(i, i + k - 1):
                if nums[j+1] != nums[j] + 1:
                    is_consecutive = False
                    break
            
            if is_consecutive:
                # The power is the last element in this subarray
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        
        return results