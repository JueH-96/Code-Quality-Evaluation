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
        palindrome_count = 0
        for length in word_lengths:
            if length % 2 == 0:
                required_pairs = length // 2
                if total_pairs >= required_pairs:
                    total_pairs -= required_pairs
                    palindrome_count += 1
            else:
                required_pairs = length // 2
                if total_pairs >= required_pairs and total_singles >= 1:
                    total_pairs -= required_pairs
                    total_singles -= 1
                    palindrome_count += 1
                    
        return palindrome_count