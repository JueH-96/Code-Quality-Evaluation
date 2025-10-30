class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count frequency of all characters
        char_count = {}
        for word in words:
            for char in word:
                char_count[char] = char_count.get(char, 0) + 1
        
        # Count pairs and singles
        pairs = 0
        singles = 0
        for count in char_count.values():
            pairs += count // 2
            singles += count % 2
        
        # Sort words by length (shortest first)
        word_lengths = sorted([len(word) for word in words])
        
        # Try to form palindromes greedily
        palindromes = 0
        
        for length in word_lengths:
            pairs_needed = length // 2
            
            if pairs >= pairs_needed:
                # We have enough pairs
                pairs -= pairs_needed
                
                if length % 2 == 1:
                    # Odd length needs a single character in middle
                    if singles > 0:
                        singles -= 1
                    elif pairs > 0:
                        # Use one character from a pair
                        pairs -= 1
                        singles += 1
                    else:
                        # Can't form this palindrome
                        break
                
                palindromes += 1
            else:
                # Not enough pairs to form more palindromes
                break
        
        return palindromes