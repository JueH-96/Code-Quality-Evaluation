class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        changes = [0] * (n + 1)
        current_change = 0
        
        for i in range(n):
            if i >= k:
                current_change -= changes[i - k]
            
            nums[i] -= current_change
            
            if nums[i] < 0:
                return False
            
            if nums[i] > 0:
                if i + k > n:
                    return False
                changes[i] += nums[i]
                current_change += nums[i]
        
        return True