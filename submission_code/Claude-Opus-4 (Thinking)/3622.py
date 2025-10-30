class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # Collect all candidate target values
        candidates = set()
        for num in nums:
            candidates.add(num)
            candidates.add(num - k)
            candidates.add(num + k)
        
        max_freq = 0
        
        for target in candidates:
            already_at_target = 0
            can_reach_target = 0
            
            for num in nums:
                if num == target:
                    already_at_target += 1
                    can_reach_target += 1
                elif abs(num - target) <= k:
                    can_reach_target += 1
            
            # We can change at most numOperations elements that aren't already at target
            achievable_freq = already_at_target + min(can_reach_target - already_at_target, numOperations)
            max_freq = max(max_freq, achievable_freq)
        
        return max_freq