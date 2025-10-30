from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # Helper function that checks if we can transform 'num' into 'target'
        # by performing at most one swap between any two digits.
        def can_transform(num: int, target: int) -> bool:
            s = list(str(num))
            # Check the case with no swap (0 operations)
            if int("".join(s)) == target:
                return True
            n = len(s)
            # Try every pair of distinct indices and swap the digits
            for i in range(n):
                for j in range(i + 1, n):
                    s[i], s[j] = s[j], s[i] 
                    if int("".join(s)) == target:
                        return True
                    s[i], s[j] = s[j], s[i]  # swap back to restore original configuration
            return False
        
        count = 0
        n = len(nums)
        # Evaluate every pair (i, j) with i < j.
        for i in range(n):
            for j in range(i + 1, n):
                # If they are already equal, no swap operation is needed.
                if nums[i] == nums[j]:
                    count += 1
                # Otherwise, check if a one-swap transformation on either number can match the other.
                else:
                    if can_transform(nums[i], nums[j]) or can_transform(nums[j], nums[i]):
                        count += 1
        return count