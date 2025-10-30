class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        """
        Calculates the total number of substrings of `word` that contain every vowel
        at least once and exactly k consonants.
        
        Args:
            word: The input string.
            k: The required number of consonants.
            
        Returns:
            The total count of such substrings.
        """
        n = len(word)
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = 0

        # The constraints (n <= 250) allow for an O(n^2) solution.
        # We can iterate through all possible substrings and check if they meet the criteria.

        # Iterate through all possible starting positions of a substring
        for i in range(n):
            # For each starting position `i`, expand the substring to the right
            # by iterating with `j`.
            vowels_found = set()
            consonants_count = 0
            
            for j in range(i, n):
                # The current substring being considered is word[i:j+1]
                char = word[j]
                
                # Update the counts for the new character added to the substring
                if char in vowels:
                    vowels_found.add(char)
                else:
                    consonants_count += 1
                
                # Check if the criteria for a valid substring are met:
                # 1. Contains every vowel ('a', 'e', 'i', 'o', 'u')
                # 2. Contains exactly k consonants
                if len(vowels_found) == 5 and consonants_count == k:
                    count += 1
                
                # Optimization: if the number of consonants already exceeds k,
                # any further extension of this substring (by increasing j) will also
                # have more than k consonants. So we can stop extending from this `i`
                # and move to the next starting position.
                if consonants_count > k:
                    break
        
        return count