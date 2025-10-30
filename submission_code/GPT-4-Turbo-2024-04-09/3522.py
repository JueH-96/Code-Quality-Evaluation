class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        def is_consecutive_sorted(subarray):
            for i in range(1, len(subarray)):
                if subarray[i] != subarray[i - 1] + 1:
                    return False
            return True
        
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if is_consecutive_sorted(subarray):
                results.append(max(subarray))
            else:
                results.append(-1)
        
        return results