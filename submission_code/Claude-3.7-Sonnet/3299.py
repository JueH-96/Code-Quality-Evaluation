class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count occurrences of each number
        counter = {}
        for num in nums:
            counter[num] = counter.get(num, 0) + 1
        
        max_length = 1  # Minimum possible answer is 1
        
        # Special case for x = 1
        if 1 in counter:
            ones_count = counter[1]
            # If count is even, we need to remove one to keep pattern length odd
            if ones_count % 2 == 0:
                max_length = max(max_length, ones_count - 1)
            else:
                max_length = max(max_length, ones_count)
        
        # Check patterns for all other values of x
        for x in counter:
            if x == 1:
                continue  # Already handled
            
            # Try to build the pattern starting with x
            current_x = x
            length = 1
            
            while True:
                next_x = current_x * current_x
                # We need at least 2 occurrences of current_x (except for the middle element)
                if next_x not in counter or counter[current_x] < 2:
                    break
                
                length += 2
                current_x = next_x
            
            max_length = max(max_length, length)
        
        return max_length