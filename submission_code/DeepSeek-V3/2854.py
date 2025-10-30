class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 0:
            return 0
        # Initialize DP table
        # dp[i][first_char][last_char] represents the minimum length of the concatenated string
        # after processing the first i words, with the first character being first_char and the last character being last_char
        # Initialize with infinity
        INF = float('inf')
        # Initialize for the first word
        first_word = words[0]
        first_char = first_word[0]
        last_char = first_word[-1]
        dp = {}
        dp[(first_char, last_char)] = len(first_word)
        for i in range(1, n):
            current_word = words[i]
            current_first_char = current_word[0]
            current_last_char = current_word[-1]
            current_length = len(current_word)
            new_dp = {}
            for (prev_first, prev_last), prev_length in dp.items():
                # Option 1: append current_word to the previous string
                # new_first_char remains prev_first
                # new_last_char is current_last_char
                # if prev_last == current_first_char, we delete one character
                new_length = prev_length + current_length
                if prev_last == current_first_char:
                    new_length -= 1
                key = (prev_first, current_last_char)
                if key in new_dp:
                    if new_length < new_dp[key]:
                        new_dp[key] = new_length
                else:
                    new_dp[key] = new_length
                # Option 2: prepend current_word to the previous string
                # new_first_char is current_first_char
                # new_last_char remains prev_last
                # if current_last_char == prev_first, we delete one character
                new_length = current_length + prev_length
                if current_last_char == prev_first:
                    new_length -= 1
                key = (current_first_char, prev_last)
                if key in new_dp:
                    if new_length < new_dp[key]:
                        new_dp[key] = new_length
                else:
                    new_dp[key] = new_length
            dp = new_dp
        # Find the minimum length in the final dp
        min_length = INF
        for key, length in dp.items():
            if length < min_length:
                min_length = length
        return min_length