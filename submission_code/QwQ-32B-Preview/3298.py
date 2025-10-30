class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Create frequency map
        freq = defaultdict(set)
        for idx, num in enumerate(nums):
            freq[num].add(idx)
            freq[num + 1].add(idx)
        
        # Extract sorted unique numbers
        sorted_nums = sorted(freq.keys())
        
        max_length = 0
        left = 0
        current_original_elements = set()
        
        for right in range(len(sorted_nums)):
            # Add original elements for the new number
            current_original_elements.update(freq[sorted_nums[right]])
            
            # Ensure numbers are consecutive in the window
            while left < right and sorted_nums[right] != sorted_nums[right - 1] + 1:
                # Remove original elements for the number at left
                for idx in freq[sorted_nums[left]]:
                    if len([num for num in sorted_nums[left:right+1] if idx in freq[num]]) == 1:
                        current_original_elements.discard(idx)
                left += 1
            
            # Update max_length if the number of distinct original elements is at least the window size
            window_size = right - left + 1
            if len(current_original_elements) >= window_size:
                max_length = max(max_length, window_size)
        
        return max_length