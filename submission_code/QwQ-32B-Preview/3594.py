from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        S = sum(nums)
        freq = Counter(nums)
        potential_outliers = []
        
        for num in freq:
            potential_outlier = S - 2 * num
            if potential_outlier in freq:
                if potential_outlier == num:
                    if freq[num] >= 2:
                        potential_outliers.append(potential_outlier)
                else:
                    if freq[num] >= 1 and freq[potential_outlier] >= 1:
                        potential_outliers.append(potential_outlier)
        return max(potential_outliers)