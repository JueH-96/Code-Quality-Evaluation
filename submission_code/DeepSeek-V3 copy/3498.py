from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # For each pair, we need to find a common X such that |a[i] - a[n-i-1]| = X
        # We can consider all possible X values and find the one that minimizes the number of changes
        # Since X can be up to k, we can iterate over all possible X values
        # For each X, we calculate the number of changes required for each pair to satisfy |a[i] - a[n-i-1]| = X
        # We choose the X that minimizes the total number of changes
        
        # Precompute the pairs
        pairs = []
        for i in range(n // 2):
            pairs.append((nums[i], nums[n - i - 1]))
        
        # Now, for each possible X, calculate the number of changes required
        min_changes = float('inf')
        for X in range(k + 1):
            changes = 0
            for a, b in pairs:
                if abs(a - b) != X:
                    # We need to change at least one of a or b
                    # Option 1: change a to b + X or b - X
                    # Option 2: change b to a + X or a - X
                    # We choose the option that requires the least changes
                    # Since we can change to any value in [0, k], we need to check if the target is within [0, k]
                    # For option 1:
                    target1 = b + X
                    target2 = b - X
                    valid1 = 0 <= target1 <= k
                    valid2 = 0 <= target2 <= k
                    # For option 2:
                    target3 = a + X
                    target4 = a - X
                    valid3 = 0 <= target3 <= k
                    valid4 = 0 <= target4 <= k
                    # Calculate the minimum changes
                    # For option 1:
                    if valid1 or valid2:
                        changes1 = 1
                    else:
                        changes1 = 2
                    # For option 2:
                    if valid3 or valid4:
                        changes2 = 1
                    else:
                        changes2 = 2
                    # Choose the minimum
                    changes += min(changes1, changes2)
            if changes < min_changes:
                min_changes = changes
            if min_changes == 0:
                break
        return min_changes