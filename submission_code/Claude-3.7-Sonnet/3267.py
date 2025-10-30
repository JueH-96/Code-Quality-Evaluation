class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        
        for char in set(s):
            # Find all consecutive occurrences of the current character
            consecutive_lengths = []
            current_length = 0
            
            for c in s:
                if c == char:
                    current_length += 1
                else:
                    if current_length > 0:
                        consecutive_lengths.append(current_length)
                        current_length = 0
            
            # Handle the last sequence if it exists
            if current_length > 0:
                consecutive_lengths.append(current_length)
            
            # Count substrings of each possible length
            length_count = {}
            for length in consecutive_lengths:
                for l in range(1, length + 1):
                    # For a sequence of length n, it contains (n-l+1) substrings of length l
                    length_count[l] = length_count.get(l, 0) + (length - l + 1)
            
            # Find the maximum length that occurs at least 3 times
            for length, count in length_count.items():
                if count >= 3 and length > max_length:
                    max_length = length
        
        return max_length