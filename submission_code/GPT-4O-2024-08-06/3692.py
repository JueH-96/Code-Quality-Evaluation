class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split the pattern into three parts based on '*'
        parts = p.split('*')
        if len(parts) != 3:
            return -1
        
        # Extract the parts
        start, middle, end = parts
        
        # Initialize variables to track the shortest length
        shortest_length = float('inf')
        
        # Iterate over the string to find the first part
        start_index = 0
        while start_index < len(s):
            # Find the start part
            start_pos = s.find(start, start_index)
            if start_pos == -1:
                break
            
            # Find the end part starting from the end of the start part
            end_pos = s.find(end, start_pos + len(start))
            if end_pos == -1:
                break
            
            # Check if the middle part exists between start and end parts
            middle_start = start_pos + len(start)
            middle_end = end_pos
            if middle in s[middle_start:middle_end]:
                # Calculate the length of the matching substring
                current_length = end_pos + len(end) - start_pos
                shortest_length = min(shortest_length, current_length)
            
            # Move start_index to the next character after the current start_pos
            start_index = start_pos + 1
        
        # If no valid substring was found, return -1
        if shortest_length == float('inf'):
            return -1
        
        return shortest_length