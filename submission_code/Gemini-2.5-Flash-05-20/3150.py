class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        
        min_len_found = float('inf')  # Stores the minimum length of a beautiful substring found so far
        result_string = ""            # Stores the lexicographically smallest beautiful substring of min_len_found

        # Iterate through all possible start indices
        for i in range(n):
            current_ones = 0
            # Iterate through all possible end indices for the current start index 'i'
            # The current substring is s[i:j+1]
            for j in range(i, n):
                # Count '1's in the character s[j]
                if s[j] == '1':
                    current_ones += 1
                
                # Check if the current substring has exactly k '1's
                if current_ones == k:
                    current_sub = s[i:j+1]
                    current_len = j - i + 1 # Length of the current substring

                    # If this substring is shorter than any previously found beautiful substring
                    if current_len < min_len_found:
                        min_len_found = current_len
                        result_string = current_sub
                    # If this substring has the same length as the current shortest,
                    # compare them lexicographically and update if current_sub is smaller
                    elif current_len == min_len_found:
                        if current_sub < result_string:
                            result_string = current_sub
                
                # Optimization: If the count of '1's exceeds k,
                # any longer substring starting at 'i' (by increasing 'j')
                # will also have more than k '1's (or the same count if only '0's are added).
                # Therefore, it cannot become a beautiful substring (with exactly k '1's) again.
                # We can stop extending the current substring from 'i' and move to the next 'i'.
                if current_ones > k:
                    break
        
        # If result_string is still empty, it means no beautiful substring was found.
        return result_string