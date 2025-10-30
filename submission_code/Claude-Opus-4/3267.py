class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        max_length = -1
        
        # Try each possible length from n-1 down to 1
        for length in range(n - 1, 0, -1):
            # Count occurrences of each special substring of this length
            count = {}
            
            # Generate all substrings of current length
            for i in range(n - length + 1):
                substring = s[i:i + length]
                
                # Check if it's a special substring (all same character)
                if len(set(substring)) == 1:
                    count[substring] = count.get(substring, 0) + 1
                    
                    # If we found one that occurs at least 3 times
                    if count[substring] >= 3:
                        return length
        
        return -1