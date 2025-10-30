class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character across all words
        char_counts = {}
        for word in words:
            for char in word:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate total pairs and odd characters available
        total_pairs = sum(count // 2 for count in char_counts.values())
        total_odds = sum(count % 2 for count in char_counts.values())
        
        # Sort words by length (to maximize number of palindromes)
        words.sort(key=len)
        
        # Try to make as many palindromes as possible
        palindromes = 0
        for word in words:
            pairs_needed = len(word) // 2
            odd_needed = len(word) % 2
            
            # Check if we have enough pairs
            if pairs_needed <= total_pairs:
                total_pairs -= pairs_needed
                
                # Handle odd-length words
                if odd_needed == 1:
                    if total_odds > 0:
                        total_odds -= 1
                    elif total_pairs > 0:
                        # Break a pair to get an odd character
                        total_pairs -= 1
                        total_odds += 1  # Breaking a pair gives us 2 odd chars
                        total_odds -= 1  # Use one for current word
                    else:
                        break
                palindromes += 1
            else:
                break
        
        return palindromes