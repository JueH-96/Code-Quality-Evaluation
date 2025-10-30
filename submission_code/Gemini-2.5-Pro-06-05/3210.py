class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        """
        Calculates the number of beautiful substrings in a given string s.

        A substring is beautiful if:
        1. The number of vowels equals the number of consonants.
        2. The product of the number of vowels and consonants is divisible by k.
        """
        n = len(s)
        vowel_set = {'a', 'e', 'i', 'o', 'u'}
        count = 0

        # Iterate through all possible starting indices of a substring.
        for i in range(n):
            vowels = 0
            consonants = 0
            # Iterate through all possible ending indices, extending the substring from i.
            for j in range(i, n):
                # Update the counts for the current substring s[i..j].
                if s[j] in vowel_set:
                    vowels += 1
                else:
                    consonants += 1
                
                # Check the conditions for a beautiful substring.
                # A non-empty substring with vowels == consonants implies vowels >= 1.
                if vowels == consonants:
                    # Condition 1: vowels == consonants is met.
                    # Now check Condition 2: (vowels * consonants) % k == 0.
                    # Since vowels == consonants, this is equivalent to (vowels * vowels) % k == 0.
                    if (vowels * consonants) % k == 0:
                        count += 1
        
        return count