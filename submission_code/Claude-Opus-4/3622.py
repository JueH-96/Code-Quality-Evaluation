class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        from collections import Counter
        
        # Count frequency of each number
        freq = Counter(nums)
        
        # Get all unique values and potential targets
        # Potential targets include all current values and values ± k from them
        targets = set()
        for num in nums:
            for delta in range(-k, k + 1):
                if num + delta > 0:  # Keep positive values only
                    targets.add(num + delta)
        
        max_freq = 0
        
        # For each potential target value
        for target in targets:
            # Count elements already at target
            current_count = freq.get(target, 0)
            
            # Count elements that can be changed to target
            changeable = 0
            for num in nums:
                if num != target and abs(num - target) <= k:
                    changeable += 1
            
            # Maximum frequency is current + min(changeable, operations left)
            total_freq = current_count + min(changeable, numOperations)
            max_freq = max(max_freq, total_freq)
        
        return max_freq