class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        from collections import Counter
        
        # Count frequency of all characters
        char_count = Counter()
        for word in words:
            for char in word:
                char_count[char] += 1
        
        # Calculate total pairs available
        pairs = sum(count // 2 for count in char_count.values())
        
        # Sort words by length (greedy: try shorter words first)
        word_lengths = sorted(len(word) for word in words)
        
        palindromes = 0
        
        for length in word_lengths:
            # For a palindrome of given length, we need length//2 pairs
            pairs_needed = length // 2
            
            if pairs >= pairs_needed:
                pairs -= pairs_needed
                palindromes += 1
            else:
                break
        
        return palindromes