class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        def join(s1, s2):
            # Returns the length after joining s1 and s2
            # s1 and s2 are represented as (first_char, last_char, length)
            first1, last1, len1 = s1
            first2, last2, len2 = s2
            
            if last1 == first2:
                return (first1, last2, len1 + len2 - 1)
            else:
                return (first1, last2, len1 + len2)
        
        # Memoization: (index, first_char, last_char) -> min_length
        memo = {}
        
        def dp(index, first_char, last_char, current_length):
            if index == len(words):
                return current_length
            
            if (index, first_char, last_char) in memo:
                return memo[(index, first_char, last_char)]
            
            next_word = words[index]
            next_first = next_word[0]
            next_last = next_word[-1]
            next_len = len(next_word)
            
            # Option 1: join(current, next_word)
            if last_char == next_first:
                option1 = dp(index + 1, first_char, next_last, current_length + next_len - 1)
            else:
                option1 = dp(index + 1, first_char, next_last, current_length + next_len)
            
            # Option 2: join(next_word, current)
            if next_last == first_char:
                option2 = dp(index + 1, next_first, last_char, current_length + next_len - 1)
            else:
                option2 = dp(index + 1, next_first, last_char, current_length + next_len)
            
            result = min(option1, option2)
            memo[(index, first_char, last_char)] = result
            return result
        
        # Start with the first word
        first_word = words[0]
        return dp(1, first_word[0], first_word[-1], len(first_word))