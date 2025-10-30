class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count frequencies of each character
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - 97] += 1
        
        # Number of full removal rounds is the maximum frequency
        R = max(freq)
        
        # Record the last occurrence index of each character
        last_occ = [-1] * 26
        for i, ch in enumerate(s):
            last_occ[ord(ch) - 97] = i
        
        # Collect characters whose frequency equals R,
        # along with their last occurrence index
        res = []
        for c in range(26):
            if freq[c] == R:
                res.append((last_occ[c], chr(c + 97)))
        
        # Sort by their position in the original string
        res.sort()
        
        # Build the answer from sorted characters
        return "".join(ch for _, ch in res)