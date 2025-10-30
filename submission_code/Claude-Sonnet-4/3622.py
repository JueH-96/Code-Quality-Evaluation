class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        from collections import defaultdict
        
        # Count original frequencies
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        # Get all unique values and sort them
        unique_nums = sorted(set(nums))
        
        max_freq = 0
        
        # For each possible target value
        for target in unique_nums:
            # Count how many numbers can contribute to this target
            already_equal = count[target]
            can_transform = []
            
            # Find all numbers that can be transformed to target
            for num in unique_nums:
                if num != target and abs(num - target) <= k:
                    can_transform.extend([num] * count[num])
            
            # We can transform at most numOperations elements
            total_freq = already_equal + min(len(can_transform), numOperations)
            max_freq = max(max_freq, total_freq)
        
        # Also consider targets that are not in the original array
        # but can be reached by transforming existing numbers
        all_possible_targets = set()
        for num in unique_nums:
            for offset in range(-k, k + 1):
                all_possible_targets.add(num + offset)
        
        for target in all_possible_targets:
            if target in count:
                continue  # Already processed
                
            can_transform = []
            for num in unique_nums:
                if abs(num - target) <= k:
                    can_transform.extend([num] * count[num])
            
            total_freq = min(len(can_transform), numOperations)
            max_freq = max(max_freq, total_freq)
        
        return max_freq