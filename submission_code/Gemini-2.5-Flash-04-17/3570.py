class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        vowels_set = {'a', 'e', 'i', 'o', 'u'}

        # Iterate through all possible start indices
        for i in range(n):
            # Reset state for each new starting position
            current_vowels = set()
            current_consonant_count = 0

            # Iterate through all possible end indices starting from the current start index
            for j in range(i, n):
                char = word[j]

                # Check if the current character is a vowel or a consonant
                if char in vowels_set:
                    current_vowels.add(char)
                else: # It's a consonant (lowercase English letter that is not a vowel)
                    current_consonant_count += 1

                # Check the conditions for the substring word[i:j+1]
                # Condition 1: Contains all 5 vowels?
                # Check if the size of the set of unique vowels found is 5.
                # This implies each of the 5 vowels is present at least once.
                has_all_vowels = len(current_vowels) == 5

                # Condition 2: Contains exactly k consonants?
                has_k_consonants = current_consonant_count == k

                # If both conditions are met for the current substring word[i:j+1], increment the total count
                if has_all_vowels and has_k_consonants:
                    count += 1

        return count