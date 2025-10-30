class Solution:
    def getSmallestString(self, s: str) -> str:
        # The lexicographically smallest string can be the original s
        best = s
        
        # Convert to list for easier swap manipulation
        n = len(s)
        
        # Consider every possible adjacent pair
        for i in range(n - 1):
            # Check if the two adjacent digits share the same parity.
            if (int(s[i]) % 2) == (int(s[i + 1]) % 2):
                # Create a candidate by swapping s[i] and s[i+1]
                candidate = list(s)
                candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
                candidate_str = "".join(candidate)
                # Update best if the candidate is lexicographically smaller.
                if candidate_str < best:
                    best = candidate_str
        
        return best