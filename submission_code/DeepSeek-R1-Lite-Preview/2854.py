class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # Map characters to indices: 'a' -> 0, 'b' -> 1, ..., 'z' -> 25
        char_to_idx = {chr(ord('a') + i): i for i in range(26)}
        
        # Initialize DP table: dp[first][last] = minimal length
        dp = [[float('inf')] * 26 for _ in range(26)]
        
        # First word
        first_word = words[0]
        f = char_to_idx[first_word[0]]
        l = char_to_idx[first_word[-1]]
        length = len(first_word)
        dp[f][l] = length
        
        # Process remaining words
        for word in words[1:]:
            new_dp = [[float('inf')] * 26 for _ in range(26)]
            fw_f = char_to_idx[word[0]]
            fw_l = char_to_idx[word[-1]]
            fw_len = len(word)
            
            for prev_f in range(26):
                for prev_l in range(26):
                    if dp[prev_f][prev_l] == float('inf'):
                        continue
                    
                    # Option 1: append the new word to the current string
                    if prev_l == fw_f:
                        new_length = dp[prev_f][prev_l] + fw_len - 1
                    else:
                        new_length = dp[prev_f][prev_l] + fw_len
                    new_state_f = prev_f
                    new_state_l = fw_l
                    new_dp[new_state_f][new_state_l] = min(new_dp[new_state_f][new_state_l], new_length)
                    
                    # Option 2: prepend the new word to the current string
                    if fw_l == prev_f:
                        new_length_prep = fw_len + dp[prev_f][prev_l] - 1
                    else:
                        new_length_prep = fw_len + dp[prev_f][prev_l]
                    new_state_f_prep = fw_f
                    new_state_l_prep = prev_l
                    new_dp[new_state_f_prep][new_state_l_prep] = min(new_dp[new_state_f_prep][new_state_l_prep], new_length_prep)
            
            dp = new_dp
        
        # Find the minimal length among all possible states
        min_length = min(min(row) for row in dp)
        return min_length