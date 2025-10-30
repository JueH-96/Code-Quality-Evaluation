class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        from functools import lru_cache
        
        n = len(words)
        
        @lru_cache(None)
        def dp(index, first_char, last_char):
            # Base case: we've processed all words
            if index == n:
                return 0
            
            current_word = words[index]
            current_first = current_word[0]
            current_last = current_word[-1]
            current_len = len(current_word)
            
            # Option 1: append current word to the result (result + current)
            append_len = current_len
            if last_char == current_first:
                append_len -= 1  # One character is removed
            append_total = append_len + dp(index + 1, first_char, current_last)
            
            # Option 2: prepend current word to the result (current + result)
            prepend_len = current_len
            if current_last == first_char:
                prepend_len -= 1  # One character is removed
            prepend_total = prepend_len + dp(index + 1, current_first, last_char)
            
            return min(append_total, prepend_total)
        
        # Start with the first word
        first_word = words[0]
        initial_len = len(first_word)
        
        # Calculate minimum length starting from index 1
        min_additional = dp(1, first_word[0], first_word[-1])
        
        return initial_len + min_additional