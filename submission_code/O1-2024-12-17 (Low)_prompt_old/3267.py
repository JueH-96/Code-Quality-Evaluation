class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        res = -1  # Default if no special substring occurs thrice
        
        # Check lengths from largest to smallest
        for length in range(n, 0, -1):
            # If we already found a valid answer larger than or equal to this length, stop early
            if res != -1 and res >= length:
                break
            
            # Try all lowercase letters as the repeating character
            for ch in "abcdefghijklmnopqrstuvwxyz":
                substring = ch * length
                count = 0
                # Count how many times this substring appears
                for start in range(n - length + 1):
                    if s[start:start+length] == substring:
                        count += 1
                    if count >= 3:
                        res = max(res, length)
                        break
                if count >= 3:
                    # No need to check other characters once we found a match for this length
                    break
        
        return res