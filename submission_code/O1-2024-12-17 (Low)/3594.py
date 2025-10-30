class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        
        total_sum = sum(nums)
        freq = Counter(nums)
        candidate = None
        
        for x in nums:
            diff = total_sum - x
            # We need (total_sum - x) to be even so it can be 2*s
            if diff % 2 != 0:
                continue
            s = diff // 2
            
            # If x == s, we need at least two occurrences of s in nums
            if x == s:
                if freq[s] > 1:
                    if candidate is None or x > candidate:
                        candidate = x
            else:
                # Otherwise, we just need s to appear at least once
                if freq[s] > 0:
                    if candidate is None or x > candidate:
                        candidate = x
                        
        return candidate