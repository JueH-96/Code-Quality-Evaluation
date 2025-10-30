class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        if nums[0] == 1 and nums[-1] == n:
            return 0
        
        first_pos = -1
        last_pos = -1
        
        for i in range(n):
            if nums[i] == 1:
                first_pos = i
            if nums[i] == n:
                last_pos = i
        
        ans = first_pos
        ans += (n - 1 - last_pos)
        
        if first_pos > last_pos:
            ans -= 1
        
        return ans