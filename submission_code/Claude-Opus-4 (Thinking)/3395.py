class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        
        for length in range(1, n + 1):
            if n % length == 0:
                # Check if all parts of length 'length' are anagrams
                first_part_sorted = sorted(s[:length])
                is_valid = True
                
                for i in range(length, n, length):
                    if sorted(s[i:i+length]) != first_part_sorted:
                        is_valid = False
                        break
                
                if is_valid:
                    return length
        
        return n