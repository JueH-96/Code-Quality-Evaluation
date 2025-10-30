from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if n < 2:
            return -1  # As per constraints, n >= 2, but handle just in case
        
        # Transform the possible array into 1 and -1
        transformed = [1 if x == 1 else -1 for x in possible]
        
        # Compute prefix sums
        prefix = [0] * n
        prefix[0] = transformed[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + transformed[i]
        
        total_sum = prefix[-1]
        
        # Check all possible k from 1 to n-1 (since Bob must play at least 1 level)
        for k in range(1, n):
            alice_sum = prefix[k-1]
            if 2 * alice_sum > total_sum:
                return k
        
        return -1