class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set("aeiou")
        n = len(word)
        result = 0
        # Traverse all possible substrings by start index
        for start in range(n):
            consonants = 0
            found = set()
            # Expand substring from 'start' ending at 'end'
            for end in range(start, n):
                c = word[end]
                if c in vowels:
                    found.add(c)
                else:
                    consonants += 1
                # If number of consonants exceeds k, break early
                if consonants > k:
                    break
                # Check if we have exactly k consonants and all vowels appeared
                if consonants == k and len(found) == 5:
                    result += 1
        return result