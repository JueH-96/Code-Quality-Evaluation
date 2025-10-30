class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        import bisect
        
        if k == 0 or numOperations == 0:
            from collections import Counter
            counter = Counter(nums)
            return max(counter.values())
        
        candidates = set(nums)
        for x in nums:
            candidates.add(x + k)
            candidates.add(x - k)
        
        sorted_nums = sorted(nums)
        
        max_freq = 0
        for target in candidates:
            # Count elements equal to target
            left = bisect.bisect_left(sorted_nums, target)
            right = bisect.bisect_right(sorted_nums, target)
            already_equal = right - left
            
            # Count elements in [target - k, target + k]
            left_range = bisect.bisect_left(sorted_nums, target - k)
            right_range = bisect.bisect_right(sorted_nums, target + k)
            total_in_range = right_range - left_range
            
            can_be_equal = total_in_range - already_equal
            
            freq = already_equal + min(numOperations, can_be_equal)
            max_freq = max(max_freq, freq)
        
        return max_freq