import math # Not strictly needed for this implementation, but good practice if math ops were used.

class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        """
        Counts the number of beautiful substrings in s.
        A substring is beautiful if:
        1. The number of vowels equals the number of consonants (vowels == consonants).
        2. The product of the number of vowels and consonants is divisible by k 
           ((vowels * consonants) % k == 0).
        
        Args:
            s: The input string consisting of lowercase English letters.
            k: A positive integer divisor.

        Returns:
            The total number of non-empty beautiful substrings in s.
        """
        n = len(s)
        count = 0
        # Define the set of vowels for efficient checking.
        vowels_set = set(['a', 'e', 'i', 'o', 'u'])

        # Precompute prefix vowel counts to efficiently calculate the number of vowels 
        # in any substring. prefix_vowels[x] will store the number of vowels in 
        # the prefix of the string s[0...x-1].
        prefix_vowels = [0] * (n + 1)
        for idx, char in enumerate(s):
            # If the character is a vowel, increment the count from the previous prefix.
            prefix_vowels[idx + 1] = prefix_vowels[idx] + (1 if char in vowels_set else 0)

        # Iterate through all possible non-empty substrings.
        # A substring is defined by its start index 'i' and end index 'j'.
        for i in range(n):  # Start index of the substring (inclusive)
            for j in range(i, n):  # End index of the substring (inclusive)
                
                # The current substring under consideration is s[i:j+1].
                length = j - i + 1
                
                # Calculate the number of vowels in the substring s[i:j+1]
                # using the precomputed prefix sums.
                # vowels(s[i..j]) = vowels(s[0..j]) - vowels(s[0..i-1])
                # vowels = prefix_vowels[j+1] - prefix_vowels[i]
                vowels = prefix_vowels[j + 1] - prefix_vowels[i]
                
                # Calculate the number of consonants in the substring.
                # consonants = total length - number of vowels
                consonants = length - vowels
                
                # Check the first condition for a beautiful substring:
                # The number of vowels must equal the number of consonants.
                if vowels == consonants:
                    # If the first condition is met, proceed to check the second condition.
                    # The product of vowels and consonants must be divisible by k.
                    # Note: Since the substring is non-empty (length >= 1) and 
                    # vowels == consonants, it implies vowels = consonants = length / 2 >= 1.
                    # So, we don't need an explicit check for vowels > 0.
                    
                    product = vowels * consonants 
                    # Or equivalently: product = vowels * vowels
                    
                    if product % k == 0:
                        # If both conditions are satisfied, the substring s[i:j+1] is beautiful.
                        # Increment the count of beautiful substrings.
                        count += 1
                        
        # Return the total count of beautiful substrings found.
        return count