class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 1:
            return len(words[0])
        dp = {}
        first_word = words[0]
        start_char = first_word[0]
        end_char = first_word[-1]
        dp[(start_char, end_char)] = len(first_word)
        
        for i in range(1, n):
            current_word = words[i]
            current_start_char = current_word[0]
            current_end_char = current_word[-1]
            current_len = len(current_word)
            next_dp = {}
            for prev_state, prev_len in dp.items():
                prev_start_char, prev_end_char = prev_state
                
                # join(str_{i-1}, words[i])
                join_len_forward = prev_len + current_len
                if prev_end_char == current_start_char:
                    join_len_forward -= 1
                next_state_forward = (prev_start_char, current_end_char)
                if next_state_forward not in next_dp or join_len_forward < next_dp[next_state_forward]:
                    next_dp[next_state_forward] = join_len_forward
                    
                # join(words[i], str_{i-1})
                join_len_backward = current_len + prev_len
                if current_end_char == prev_start_char:
                    join_len_backward -= 1
                next_state_backward = (current_start_char, prev_end_char)
                if next_state_backward not in next_dp or join_len_backward < next_dp[next_state_backward]:
                    next_dp[next_state_backward] = join_len_backward
                    
            dp = next_dp
            
        min_len = float('inf')
        for length in dp.values():
            min_len = min(min_len, length)
            
        return min_len