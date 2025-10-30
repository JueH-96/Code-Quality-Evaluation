class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
        n = len(word)
        count = 0
        
        for start in range(n):
            freq = [0]*5
            consonants = 0
            
            for end in range(start, n):
                ch = word[end]
                if ch in vowels:
                    freq[vowels[ch]] += 1
                else:
                    consonants += 1
                
                # If we exceed k consonants, no need to expand further for this start
                if consonants > k:
                    break
                
                # Check if we have exactly k consonants and all vowels at least once
                if consonants == k and all(x > 0 for x in freq):
                    count += 1
        
        return count