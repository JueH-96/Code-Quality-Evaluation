class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        from collections import defaultdict
        
        def is_valid(counts):
            # Check if all characters have exactly k occurrences
            for v in counts.values():
                if v != k:
                    return False
            return True
        
        def max_diff_two(chars):
            # Check if the difference between any two adjacent characters is at most 2
            sorted_chars = sorted(chars)
            for i in range(1, len(sorted_chars)):
                if ord(sorted_chars[i]) - ord(sorted_chars[i-1]) > 2:
                    return False
            return True
        
        n = len(word)
        result = 0
        
        # Generate all possible substrings
        for start in range(n):
            counts = defaultdict(int)
            unique_chars = set()
            
            for end in range(start, n):
                char = word[end]
                counts[char] += 1
                unique_chars.add(char)
                
                # Check if current substring is valid
                if is_valid(counts) and max_diff_two(unique_chars):
                    result += 1
        
        return result