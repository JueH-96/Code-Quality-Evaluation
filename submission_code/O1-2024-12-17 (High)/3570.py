class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        result = 0
        
        for start in range(n):
            freq = {v: 0 for v in vowels}  # to track frequency of each vowel in current substring
            distinct_vowels = 0          # how many different vowels are present at least once
            consonants = 0               # number of consonants in the current substring
            
            for end in range(start, n):
                c = word[end]
                if c in vowels:
                    freq[c] += 1
                    # If this is the first time this vowel appears, increment distinct_vowels
                    if freq[c] == 1:
                        distinct_vowels += 1
                else:
                    consonants += 1
                    
                # If we exceed the allowed number of consonants, stop expanding
                if consonants > k:
                    break
                
                # Check if we have all vowels and exactly k consonants
                if distinct_vowels == 5 and consonants == k:
                    result += 1
                    
        return result