class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        
        # Try all divisors of n starting from 1
        for length in range(1, n + 1):
            if n % length == 0:
                # Check if all chunks of this length are anagrams
                if self.isValidLength(s, length):
                    return length
        
        return n  # This should never be reached given the constraints
    
    def isValidLength(self, s: str, length: int) -> bool:
        # Get the character count of the first chunk
        first_chunk_count = {}
        for i in range(length):
            char = s[i]
            first_chunk_count[char] = first_chunk_count.get(char, 0) + 1
        
        # Check all other chunks
        num_chunks = len(s) // length
        for chunk_idx in range(1, num_chunks):
            chunk_count = {}
            start = chunk_idx * length
            end = start + length
            
            # Count characters in current chunk
            for i in range(start, end):
                char = s[i]
                chunk_count[char] = chunk_count.get(char, 0) + 1
            
            # Compare with first chunk
            if chunk_count != first_chunk_count:
                return False
        
        return True