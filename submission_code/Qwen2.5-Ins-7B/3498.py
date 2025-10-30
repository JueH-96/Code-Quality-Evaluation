from collections import Counter
from itertools import combinations

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = [0] * k
        right = [0] * k
        pairs = Counter()
        
        # Count the frequency of each pair (nums[i], nums[n-i-1])
        for i in range(k):
            left[i] = nums[i]
            right[i] = nums[n - i - 1]
            pairs[(left[i], right[i])] += 1
        
        # Calculate the maximum number of pairs that can be matched
        max_pairs = max(pairs.values(), default=0)
        
        # Calculate the minimum number of changes required
        min_changes = n - max_pairs
        
        # Try to improve the result by considering all possible pairs
        for i in range(k):
            for j in range(k):
                if (left[i], right[j]) in pairs:
                    pairs[(left[i], right[j])] -= 1
                    if pairs[(left[i], right[j])] == 0:
                        del pairs[(left[i], right[j])]
                    min_changes = min(min_changes, n - max(pairs.values(), default=0))
                    pairs[(left[i], right[j])] += 1
        
        return min_changes