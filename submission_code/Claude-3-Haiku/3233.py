from collections import Counter

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        max_partitions = 0
        
        # Try changing each character in the string
        for i in range(n):
            # Create a copy of the string
            s_copy = list(s)
            
            # Change the current character to a different lowercase letter
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != s_copy[i]:
                    s_copy[i] = c
                    s_copy = ''.join(s_copy)
                    
                    # Count the number of partitions
                    partitions = 0
                    distinct_chars = set()
                    left = 0
                    for right in range(n):
                        distinct_chars.add(s_copy[right])
                        while len(distinct_chars) > k:
                            distinct_chars.remove(s_copy[left])
                            left += 1
                        partitions = max(partitions, right - left + 1)
                    
                    max_partitions = max(max_partitions, partitions)
                    
                    # Restore the original character
                    s_copy[i] = s[i]
        
        return max_partitions