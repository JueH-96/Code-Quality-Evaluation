class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        result = 0
        
        for i in range(n):
            vowel_seen = set()
            consonant_count = 0
            
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_seen.add(word[j])
                else:
                    consonant_count += 1
                
                # If consonant count exceeds k, no point in expanding further
                if consonant_count > k:
                    break
                
                # Check if we have all vowels and exactly k consonants
                if len(vowel_seen) == 5 and consonant_count == k:
                    result += 1
        
        return result