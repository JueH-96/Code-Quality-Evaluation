class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        def has_all_vowels(freq):
            return all(freq[c] > 0 for c in 'aeiou')
            
        def count_consonants(freq):
            return sum(freq[c] for c in freq if c not in 'aeiou')
        
        n = len(word)
        result = 0
        
        # Try all possible substrings
        for i in range(n):
            freq = {}
            for c in 'abcdefghijklmnopqrstuvwxyz':
                freq[c] = 0
                
            # Build frequency map for substring starting at i
            for j in range(i, n):
                freq[word[j]] += 1
                
                # Check if current substring has all vowels and k consonants
                if has_all_vowels(freq) and count_consonants(freq) == k:
                    result += 1
                    
        return result