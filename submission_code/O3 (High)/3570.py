class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        answer = 0
        
        for start in range(n):
            # counters re-initialised for every new start position
            vowel_seen = {v: 0 for v in vowels}
            consonants = 0
            have_all_vowels = False     # flag to avoid repeated all() checks if desired
            
            for end in range(start, n):
                ch = word[end]
                
                if ch in vowels:
                    vowel_seen[ch] += 1
                    # if we have just completed the set of vowels, update flag
                    if not have_all_vowels and all(vowel_seen[v] > 0 for v in vowels):
                        have_all_vowels = True
                else:
                    consonants += 1
                
                # once consonants exceed k, any further extension will only increase it
                if consonants > k:
                    break
                
                # valid substring if exact k consonants and all vowels present
                if consonants == k and have_all_vowels:
                    answer += 1
                    
        return answer