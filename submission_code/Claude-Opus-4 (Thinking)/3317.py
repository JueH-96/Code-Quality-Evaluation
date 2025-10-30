class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character across all strings
        char_count = {}
        for word in words:
            for char in word:
                char_count[char] = char_count.get(char, 0) + 1
        
        # Count the number of pairs we can form
        pairs = sum(count // 2 for count in char_count.values())
        
        # Sort the word lengths (greedy - form shorter palindromes first)
        lengths = sorted(len(word) for word in words)
        
        # Try to form palindromes starting from the shortest
        palindromes = 0
        for length in lengths:
            needed_pairs = length // 2
            if pairs >= needed_pairs:
                pairs -= needed_pairs
                palindromes += 1
            else:
                break
        
        return palindromes