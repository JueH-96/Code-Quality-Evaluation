class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find maximum element which would be n
        n = max(nums)
        
        # Length of base[n] should be n+1
        if len(nums) != n + 1:
            return False
            
        # Create expected base[n] array
        base = list(range(1, n)) + [n, n]
        
        # Sort both arrays and compare
        nums.sort()
        return nums == base