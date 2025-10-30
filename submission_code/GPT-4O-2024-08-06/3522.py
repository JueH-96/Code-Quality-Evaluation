class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        results = []
        
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            if self.is_consecutive_sorted(subarray):
                results.append(max(subarray))
            else:
                results.append(-1)
        
        return results
    
    def is_consecutive_sorted(self, subarray: List[int]) -> bool:
        # Check if the subarray is sorted and consecutive
        subarray_sorted = sorted(subarray)
        for j in range(1, len(subarray_sorted)):
            if subarray_sorted[j] != subarray_sorted[j-1] + 1:
                return False
        return True