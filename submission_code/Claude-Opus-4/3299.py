class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each number
        count = Counter(nums)
        
        max_length = 1  # At least we can always select one element
        
        # Special case for 1
        if 1 in count:
            # For 1, we can use all of them, but the pattern must be odd length
            # If we have even number of 1s, we use one less to make it odd
            max_length = count[1] if count[1] % 2 == 1 else count[1] - 1
        
        # Try each number as a potential base (except 1)
        for base in count:
            if base == 1:
                continue
                
            current = base
            length = 0
            
            # Build the pattern going up: x, x^2, x^4, ...
            while current in count and count[current] >= 2:
                length += 2  # We need 2 of each (except possibly the last)
                current = current * current
            
            # Check if we can add the middle element
            if current in count:
                length += 1
            else:
                # If we can't find the next power, we need to remove the last 2 we added
                # and add 1 for the middle element
                if length > 0:
                    length = length - 2 + 1
            
            max_length = max(max_length, length)
        
        return max_length