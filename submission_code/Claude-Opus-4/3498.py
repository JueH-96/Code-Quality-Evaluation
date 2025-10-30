class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = n // 2
        
        # For each possible X value, count how many changes are needed
        # X can range from 0 to k
        changes_needed = [0] * (k + 2)  # k+2 to handle boundary
        
        for i in range(pairs):
            a, b = nums[i], nums[n - i - 1]
            
            # Current difference without any change
            current_diff = abs(a - b)
            
            # Maximum X achievable with at most 1 change
            # We can change one number to any value in [0, k]
            max_with_one_change = max(max(a, b), k - min(a, b))
            
            # For X = 0 to k:
            # - If X == current_diff: 0 changes needed
            # - If X <= max_with_one_change: 1 change needed
            # - Otherwise: 2 changes needed
            
            # Add 2 changes for all X values initially
            changes_needed[0] += 2
            
            # For X values that can be achieved with 1 change, reduce by 1
            if max_with_one_change + 1 <= k + 1:
                changes_needed[max_with_one_change + 1] -= 1
            
            # For the exact current difference, reduce by another 1 (total 0 changes)
            changes_needed[current_diff] += 1
            if current_diff + 1 <= k + 1:
                changes_needed[current_diff + 1] -= 1
        
        # Calculate prefix sum to get actual changes needed for each X
        min_changes = float('inf')
        current_changes = 0
        
        for x in range(k + 1):
            current_changes += changes_needed[x]
            min_changes = min(min_changes, current_changes)
        
        return min_changes