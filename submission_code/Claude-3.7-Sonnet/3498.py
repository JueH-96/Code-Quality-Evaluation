class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_changes = float('inf')
        
        # Try each possible value of X from 0 to k
        for X in range(0, k + 1):
            total_changes = 0
            
            # Check each pair of elements
            for i in range(n // 2):
                a, b = nums[i], nums[n - i - 1]
                
                # If the absolute difference is already X, no changes are needed
                if abs(a - b) == X:
                    continue
                
                # Try to change only one element
                can_change_one = (0 <= b + X <= k) or (0 <= a - X <= k) or (0 <= b - X <= k) or (0 <= a + X <= k)
                
                if can_change_one:
                    total_changes += 1
                else:
                    # If we can't make abs(a - b) = X by changing only one element, we change both
                    total_changes += 2
            
            min_changes = min(min_changes, total_changes)
        
        return min_changes