from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        freq = Counter(nums)
        total_sum = sum(nums)

        # Check potential outliers from largest to smallest
        for candidate in range(1000, -1001, -1):
            if freq[candidate] > 0:
                # Temporarily remove one occurrence of candidate
                freq[candidate] -= 1
                
                candidate_sum = total_sum - candidate
                # We need the sum to be even in order to find a valid x = candidate_sum // 2
                if candidate_sum % 2 == 0:
                    x = candidate_sum // 2
                    # Check if x exists in the remaining elements
                    if freq[x] > 0:
                        # candidate can serve as an outlier
                        freq[candidate] += 1
                        return candidate
                    
                # Restore frequency
                freq[candidate] += 1
        
        # According to the problem statement, there's always a valid outlier.
        # So we should never reach this point.
        return None