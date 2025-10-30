class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        counts = Counter(nums)
        total = sum(nums)
        max_outlier = float("-inf")
        
        for s in nums:
            # Potential outlier is total - 2*s
            o = total - 2*s
            if o in counts:
                if o == s:
                    # We need at least 2 occurrences if outlier == sum element
                    if counts[o] > 1:
                        max_outlier = max(max_outlier, o)
                else:
                    max_outlier = max(max_outlier, o)
        
        return max_outlier