from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Count the frequency of each character in the string
        char_count = Counter(s)
        
        # Initialize the minimum length to infinity
        min_length = float('inf')
        
        # Iterate over all possible lengths of the anagram
        for length in range(1, len(s) + 1):
            # Check if the length is a divisor of the string length
            if len(s) % length == 0:
                # Initialize a flag to check if the anagram is valid
                is_valid = True
                
                # Initialize a counter for the anagram
                anagram_count = Counter()
                
                # Iterate over the string in chunks of the current length
                for i in range(0, len(s), length):
                    # Get the chunk
                    chunk = s[i:i + length]
                    
                    # Count the frequency of each character in the chunk
                    chunk_count = Counter(chunk)
                    
                    # If this is the first chunk, set the anagram count
                    if i == 0:
                        anagram_count = chunk_count
                    # Otherwise, check if the chunk is an anagram of the previous chunks
                    else:
                        if chunk_count != anagram_count:
                            # If not, set the flag to False and break
                            is_valid = False
                            break
                
                # If the anagram is valid, update the minimum length
                if is_valid:
                    min_length = min(min_length, length)
        
        # Return the minimum length
        return min_length