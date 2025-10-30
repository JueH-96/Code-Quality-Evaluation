class Solution:
    def getSmallestString(self, s: str) -> str:
        # Initially the best candidate is the string itself (case of no swap)
        best = s
        n = len(s)
        
        # Convert string to list for easier swapping simulation (but we will convert back to string)
        s_list = list(s)
        
        # Iterate over each valid adjacent pair with same parity.
        for i in range(n - 1):
            # Check if the adjacent digits have the same parity using int conversion.
            if (int(s_list[i]) % 2) == (int(s_list[i+1]) % 2):
                # Create a copy of the list so that we swap without affecting original.
                candidate = s_list.copy()
                # Swap the adjacent digits.
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                # Convert back to string.
                candidate_str = "".join(candidate)
                # Update best if candidate_str is lexicographically smaller.
                if candidate_str < best:
                    best = candidate_str
        return best