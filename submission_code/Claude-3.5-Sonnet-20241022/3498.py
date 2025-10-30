class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        def check(x):
            # For a given difference x, check minimum changes needed
            changes = 0
            for i in range(n//2):
                a, b = nums[i], nums[n-1-i]
                # Find if we can make abs(a-b) = x with minimum changes
                # Case 1: a-b = x
                # Case 2: b-a = x
                valid = False
                min_changes = float('inf')
                
                # Try a-b = x
                new_b = a - x
                if 0 <= new_b <= k:
                    changes_needed = 0 if b == new_b else 1
                    min_changes = min(min_changes, changes_needed)
                    valid = True
                
                # Try b-a = x
                new_b = a + x
                if 0 <= new_b <= k:
                    changes_needed = 0 if b == new_b else 1
                    min_changes = min(min_changes, changes_needed)
                    valid = True
                    
                # Try a-b = -x
                new_a = b + x
                if 0 <= new_a <= k:
                    changes_needed = 0 if a == new_a else 1
                    min_changes = min(min_changes, changes_needed)
                    valid = True
                    
                # Try b-a = -x
                new_a = b - x
                if 0 <= new_a <= k:
                    changes_needed = 0 if a == new_a else 1
                    min_changes = min(min_changes, changes_needed)
                    valid = True
                
                if not valid:
                    return float('inf')
                changes += min_changes
                
            return changes
        
        # Try all possible differences
        min_changes = float('inf')
        for x in range(k+1):
            min_changes = min(min_changes, check(x))
            
        return min_changes if min_changes != float('inf') else -1