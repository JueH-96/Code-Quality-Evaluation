class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        vowels_set = {'a', 'e', 'i', 'o', 'u'} 

        # Iterate through all possible starting positions of a substring
        for i in range(n):
            # For each starting position i, we will extend the substring
            # character by character to the right (j).
            
            # Keep track of unique vowels found in the current substring word[i...j]
            current_vowels_present = set() 
            
            # Count of consonants in the current substring word[i...j]
            current_consonant_count = 0

            # Iterate through all possible ending positions for the substring, starting from i
            for j in range(i, n):
                char = word[j]

                if char in vowels_set:
                    current_vowels_present.add(char)
                else:
                    current_consonant_count += 1
                
                # Optimization: If the consonant count exceeds k, this substring (and any longer
                # substrings starting at 'i') will also have more than k consonants.
                # Since we need exactly k consonants, we can stop extending this substring
                # from 'i' and move to the next starting position.
                if current_consonant_count > k:
                    break # Break out of the inner loop (j loop)
                
                # Check the conditions for the substring word[i:j+1]
                # Condition 1: Contains exactly k consonants
                # Condition 2: Contains all 5 vowels ('a', 'e', 'i', 'o', 'u')
                if current_consonant_count == k and len(current_vowels_present) == 5:
                    count += 1
        
        return count