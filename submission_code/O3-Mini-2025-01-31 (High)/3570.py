class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels_set = set("aeiou")
        total = 0
        n = len(word)
        
        # Iterate over all possible starting indices of substrings.
        for i in range(n):
            consonant_count = 0
            found_vowels = set()
            # Expand the substring from index i to j.
            for j in range(i, n):
                ch = word[j]
                if ch in vowels_set:
                    found_vowels.add(ch)
                else:
                    consonant_count += 1
                # If consonants exceed k, no need to extend further.
                if consonant_count > k:
                    break
                # Check if we have exactly k consonants and all vowels are present.
                if consonant_count == k and len(found_vowels) == 5:
                    total += 1
        return total