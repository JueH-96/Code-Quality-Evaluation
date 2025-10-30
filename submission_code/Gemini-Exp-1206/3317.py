class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        counts = [0] * 26
        for word in words:
            for char in word:
                counts[ord(char) - ord('a')] += 1
        
        pairs = 0
        odd_count = 0
        for count in counts:
            pairs += count // 2
            odd_count += count % 2
        
        words.sort(key=len)
        
        result = 0
        for word in words:
            word_len = len(word)
            required_pairs = word_len // 2
            
            if pairs >= required_pairs:
                pairs -= required_pairs
                if word_len % 2 == 0 or odd_count > 0:
                    result += 1
                    odd_count = max(0, odd_count - (word_len % 2))
        
        return result