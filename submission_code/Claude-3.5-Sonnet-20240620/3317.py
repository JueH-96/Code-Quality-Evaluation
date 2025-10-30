class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character
        char_count = {}
        for word in words:
            for char in word:
                char_count[char] = char_count.get(char, 0) + 1
        
        # Count the number of pairs and single characters
        pairs = sum(count // 2 for count in char_count.values())
        singles = sum(count % 2 for count in char_count.values())
        
        # Sort word lengths in ascending order
        word_lengths = sorted(len(word) for word in words)
        
        palindromes = 0
        for length in word_lengths:
            pairs_needed = length // 2
            singles_needed = length % 2
            
            if pairs >= pairs_needed and singles >= singles_needed:
                pairs -= pairs_needed
                singles -= singles_needed
                palindromes += 1
            elif pairs >= pairs_needed and singles + (pairs - pairs_needed) * 2 >= singles_needed:
                extra_pairs = (singles_needed - singles + 1) // 2
                pairs -= (pairs_needed + extra_pairs)
                singles = singles + extra_pairs * 2 - singles_needed
                palindromes += 1
            else:
                break
        
        return palindromes