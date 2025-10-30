class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        
        # Helper function to compute character counts for a substring
        # Returns a list of 26 integers representing counts for 'a' through 'z'
        def get_counts(substring):
            counts = [0] * 26 
            for char in substring:
                counts[ord(char) - ord('a')] += 1
            return counts

        # Iterate through possible lengths L, starting from 1
        # The minimum length L must be a divisor of n because s is a concatenation
        # of N/L parts of length L.
        # We check divisors in increasing order, so the first L that works is the minimum.
        for l in range(1, n + 1):
            # Check if L is a divisor of n
            if n % l == 0:
                # L is a potential length for t.
                # Check if all substrings of length L are anagrams of each other.
                # This is done by comparing the character counts of each substring
                # with the character counts of the first substring s[0:L].
                
                # Compute character counts for the first substring s[0:L]
                base_counts = get_counts(s[0:l])
                
                is_valid = True
                num_parts = n // l # The number of equal length parts s is divided into
                
                # Compare counts for subsequent substrings s[i*L:(i+1)*L]
                # We start from i=1 because i=0 is the base_counts
                for i in range(1, num_parts):
                    start_index = i * l
                    end_index = (i + 1) * l
                    current_substring = s[start_index:end_index]
                    
                    current_counts = get_counts(current_substring)
                    
                    # Check if the current substring has the same counts as the first one
                    # List comparison works element by element.
                    if base_counts != current_counts:
                        is_valid = False
                        break # If one part doesn't match, this L is not valid
                
                # If the loop completed and is_valid is still True,
                # it means all parts were anagrams of the first part (and thus of each other).
                # Since we are iterating L in increasing order, this is the minimum such L.
                # This L is the minimum possible length of the string t.
                return l
        
        # This part should technically not be reached given the problem constraints,
        # as L=n is always a valid length (s is a concatenation of 1 anagram of s itself).
        # Including it provides a fallback, returning n if no smaller length is found.
        return n