class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        def is_strictly_increasing(start):
            return all(nums[i] < nums[i + 1] for i in range(start, start + k - 1))
        
        return any(is_strictly_increasing(a) and is_strictly_increasing(a + k) 
                   for a in range(n - 2 * k + 1))