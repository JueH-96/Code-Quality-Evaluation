class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 1:
            return len(words[0])
        dp = {}
        first_word = words[0]
        first_char = first_word[0]
        last_char = first_word[-1]
        dp[(first_char, last_char)] = len(first_word)
        
        for i in range(1, n):
            current_word = words[i]
            current_first_char = current_word[0]
            current_last_char = current_word[-1]
            current_len = len(current_word)
            next_dp = {}
            for (prev_first_char, prev_last_char), prev_len in dp.items():
                # join(str_{i-1}, words[i])
                len1 = prev_len + current_len
                if prev_last_char == current_first_char:
                    len1 -= 1
                new_last_char1 = current_last_char
                new_first_char1 = prev_first_char
                state1 = (new_first_char1, new_last_char1)
                next_dp[state1] = min(next_dp.get(state1, float('inf')), len1)
                
                # join(words[i], str_{i-1})
                len2 = current_len + prev_len
                if current_last_char == prev_first_char:
                    len2 -= 1
                new_first_char2 = current_first_char
                new_last_char2 = prev_last_char
                state2 = (new_first_char2, new_last_char2)
                next_dp[state2] = min(next_dp.get(state2, float('inf')), len2)
                
            dp = next_dp
            
        min_len = float('inf')
        for length in dp.values():
            min_len = min(min_len, length)
            
        return min_len