class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        
        # We'll iterate over possible lengths (starting from largest to smallest)
        for length in range(n, 0, -1):
            # Check each lowercase letter
            for c in "abcdefghijklmnopqrstuvwxyz":
                substring = c * length
                
                # Count occurrences with possible overlap
                count = 0
                start = 0
                while True:
                    idx = s.find(substring, start)
                    if idx == -1:
                        break
                    count += 1
                    # Move start by 1 to allow overlapping
                    start = idx + 1
                    
                # If we have at least 3 occurrences, return this length.
                if count >= 3:
                    return length
        
        # If not found, return -1
        return -1