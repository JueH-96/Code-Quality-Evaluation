class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        changes = 0
        
        for i in range(n // 2):
            a, b = nums[i], nums[n - 1 - i]
            if a == b:
                continue
            
            diff = abs(a - b)
            min_changes = float('inf')
            
            for x in range(diff, k + 1):
                c1 = max(0, x - a) + max(0, x - b)
                c2 = abs(a - (b + x)) + 1 if b + x <= k else float('inf')
                c3 = abs(b - (a + x)) + 1 if a + x <= k else float('inf')
                min_changes = min(min_changes, c1, c2, c3)
            
            changes += min_changes
        
        return changes