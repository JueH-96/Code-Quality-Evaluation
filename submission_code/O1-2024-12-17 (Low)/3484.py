class Solution:
    def getSmallestString(self, s: str) -> str:
        # Initialize the best (smallest) string as the original
        best = s
        
        # Try swapping each adjacent pair of digits that have the same parity
        for i in range(len(s) - 1):
            if (int(s[i]) % 2) == (int(s[i+1]) % 2):  # Same parity
                # Create a candidate string by swapping
                candidate_list = list(s)
                candidate_list[i], candidate_list[i+1] = candidate_list[i+1], candidate_list[i]
                candidate = ''.join(candidate_list)
                
                # Update the best string if this candidate is smaller
                if candidate < best:
                    best = candidate
                    
        return best