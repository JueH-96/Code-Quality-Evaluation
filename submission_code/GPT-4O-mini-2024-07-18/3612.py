class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        def is_strictly_increasing(subarray):
            return all(subarray[i] < subarray[i + 1] for i in range(len(subarray) - 1))
        
        n = len(nums)
        
        for i in range(n - 2 * k + 1):
            if is_strictly_increasing(nums[i:i + k]) and is_strictly_increasing(nums[i + k:i + 2 * k]):
                return True
        
        return False