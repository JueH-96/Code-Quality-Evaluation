class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        # Initialize infinite length and an empty substring as the shortest beautiful substring
        shortest_len = float('inf')
        shortest_sub = ""
        
        # Check all possible slicing points
        for i in range(len(s)):
            ones_count = 0
            slice_end = i
            # Loop forward to find the k-th '1'
            while slice_end < len(s) and ones_count < k:
                if s[slice_end] == '1':
                    ones_count += 1
                slice_end += 1
                
            # Check if we found k '1's without going out of bounds
            if ones_count == k:
                slice_end -= 1  # adjust to the last '1'
                current_slice = s[i:slice_end + 1]
                if len(current_slice) < shortest_len:
                    # Update if we found a shorter beautiful substring
                    shortest_len = len(current_slice)
                    shortest_sub = current_slice
                elif len(current_slice) == shortest_len:
                    # If same length, update if it's smaller lexicographically
                    shortest_sub = min(shortest_sub, current_slice)
        
        return shortest_sub