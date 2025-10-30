class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count frequency of each character across all words
        char_freq = {}
        for word in words:
            for c in word:
                char_freq[c] = char_freq.get(c, 0) + 1
        
        # Count number of pairs (characters that appear even times)
        pairs = sum(freq // 2 for freq in char_freq.values())
        singles = sum(freq % 2 for freq in char_freq.values())
        
        # Sort word lengths in ascending order
        word_lens = sorted(len(word) for word in words)
        
        count = 0
        # For each word length
        for length in word_lens:
            pairs_needed = length // 2
            single_needed = length % 2
            
            # If we have enough pairs for this word
            if pairs >= pairs_needed:
                pairs -= pairs_needed
                # If odd length word needs a single character
                if single_needed:
                    if singles > 0:
                        singles -= 1
                    else:
                        # Convert a pair to singles if needed
                        pairs += 1
                        singles += 1
                        pairs -= pairs_needed
                        singles -= 1
                count += 1
        
        return count