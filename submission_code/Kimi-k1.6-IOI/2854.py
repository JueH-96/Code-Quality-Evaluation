class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Initialize DP with the first word's first and last characters and its length
        dp = {}
        first_word = words[0]
        first_char = first_word[0]
        last_char = first_word[-1]
        dp[(first_char, last_char)] = len(first_word)
        
        for i in range(1, len(words)):
            current_word = words[i]
            cfirst = current_word[0]
            clast = current_word[-1]
            current_length = len(current_word)
            new_dp = {}
            
            # Iterate over all previous states
            for (f, l), length in dp.items():
                # Option 1: Append current_word to the end of the previous string
                overlap = 1 if l == cfirst else 0
                new_len = length + current_length - overlap
                new_state = (f, clast)
                if new_state in new_dp:
                    if new_len < new_dp[new_state]:
                        new_dp[new_state] = new_len
                else:
                    new_dp[new_state] = new_len
                
                # Option 2: Prepend current_word to the beginning of the previous string
                overlap = 1 if clast == f else 0
                new_len = length + current_length - overlap
                new_state = (cfirst, l)
                if new_state in new_dp:
                    if new_len < new_dp[new_state]:
                        new_dp[new_state] = new_len
                else:
                    new_dp[new_state] = new_len
            
            dp = new_dp
        
        # The minimum value among all possible states in the final DP is the answer
        return min(dp.values())