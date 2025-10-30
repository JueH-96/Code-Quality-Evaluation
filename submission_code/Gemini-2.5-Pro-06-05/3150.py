class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        """
        Finds the lexicographically smallest beautiful substring of the shortest possible length.

        A beautiful substring is one with exactly k '1's.
        """
        
        # Step 1: Find the indices of all '1's in the string.
        one_indices = [i for i, char in enumerate(s) if char == '1']
        
        # Step 2: If the total number of '1's is less than k, no beautiful substring can be formed.
        if len(one_indices) < k:
            return ""
            
        min_len = float('inf')
        result_substring = ""
        
        # Step 3: Iterate through all possible contiguous groups of k '1's.
        # A minimal beautiful substring must span from some i-th '1' to the (i+k-1)-th '1'.
        for i in range(len(one_indices) - k + 1):
            # Determine the start and end indices of the candidate substring in the original string s.
            start_idx = one_indices[i]
            end_idx = one_indices[i + k - 1]
            
            candidate = s[start_idx : end_idx + 1]
            current_len = len(candidate)
            
            # Step 4: Update the result based on length and lexicographical order.
            if current_len < min_len:
                # Found a new shortest substring.
                min_len = current_len
                result_substring = candidate
            elif current_len == min_len:
                # Found a substring of the same minimal length.
                # Keep the one that is lexicographically smaller.
                result_substring = min(result_substring, candidate)
                    
        # Step 5: Return the final result.
        return result_substring