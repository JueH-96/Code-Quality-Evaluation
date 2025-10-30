class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # This variable will store the total count of substrings satisfying the k-constraint.
        valid_substrings_count = 0
        
        # Outer loop iterates through all possible starting indices of a substring.
        for i in range(n):
            # For each starting index i, we initialize counts of zeros and ones for substrings starting at i.
            zeros_in_current_substring = 0
            ones_in_current_substring = 0
            
            # Inner loop iterates through all possible ending indices of a substring,
            # given that the substring starts at index i.
            # The character s[j] is the newest character added to the substring s[i..j].
            for j in range(i, n):
                # Update counts based on the character s[j].
                if s[j] == '0':
                    zeros_in_current_substring += 1
                else:  # s[j] must be '1' as per problem constraints (binary string).
                    ones_in_current_substring += 1
                
                # Now, zeros_in_current_substring and ones_in_current_substring
                # are the counts for the substring s[i..j].
                # Check if this substring satisfies the k-constraint.
                # The k-constraint is satisfied if:
                #   (number of 0's <= k) OR (number of 1's <= k)
                if zeros_in_current_substring <= k or ones_in_current_substring <= k:
                    valid_substrings_count += 1
        
        return valid_substrings_count