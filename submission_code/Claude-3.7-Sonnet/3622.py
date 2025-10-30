class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        from collections import Counter
        
        # Count frequency of each element
        count = Counter(nums)
        
        max_frequency = 0
        
        # For each unique value in the array, consider it as a target
        for target in count:
            # Start with the current frequency of this target
            current_frequency = count[target]
            
            # Count how many other elements can be adjusted to become the target
            adjustable_count = 0
            for num, freq in count.items():
                if num != target:
                    # Check if we can add a value in [-k, k] to make num equal to target
                    adjustment = target - num
                    if -k <= adjustment <= k:
                        adjustable_count += freq
            
            # We can adjust at most numOperations elements
            final_frequency = current_frequency + min(numOperations, adjustable_count)
            
            # Update the maximum frequency we've seen
            max_frequency = max(max_frequency, final_frequency)
        
        return max_frequency