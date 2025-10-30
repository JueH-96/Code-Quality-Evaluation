class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        # Create a frequency counter for the original numbers
        count = Counter(nums)
        max_beauty = 0
        
        # Iterate through each unique number in the counter
        for num in count:
            # Calculate the range of values we can convert to
            low = num - k
            high = num + k
            
            # Calculate the total count of numbers that can be converted to this range
            total_count = 0
            
            for key in count:
                if low <= key <= high:
                    total_count += count[key]
            
            # Update the maximum beauty found
            max_beauty = max(max_beauty, total_count)
        
        return max_beauty