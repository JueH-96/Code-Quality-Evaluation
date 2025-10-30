from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        char_counts = Counter()
        word_lengths = []
        for word in words:
            word_lengths.append(len(word))
            for char in word:
                char_counts[char] += 1
        
        total_pairs = 0
        total_singles = 0
        for char in char_counts:
            total_pairs += char_counts[char] // 2
            total_singles += char_counts[char] % 2
            
        word_lengths.sort()
        n = len(word_lengths)
        max_palindromes = 0
        
        for k in range(n + 1):
            pairs_needed = 0
            odd_length_words = 0
            for i in range(k):
                pairs_needed += word_lengths[i] // 2
                if word_lengths[i] % 2 != 0:
                    odd_length_words += 1
                    
            if pairs_needed <= total_pairs and odd_length_words <= total_singles:
                max_palindromes = max(max_palindromes, k)
                
        return max_palindromes