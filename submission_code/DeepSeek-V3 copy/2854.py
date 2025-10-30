class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 0:
            return 0
        # Initialize DP table
        # dp[i][first_char][last_char] represents the minimum length of the concatenated string up to the i-th word, with the first character being first_char and the last character being last_char
        # Initialize with the first word
        first_word = words[0]
        first_char = first_word[0]
        last_char = first_word[-1]
        length = len(first_word)
        # Initialize the DP table
        # We will use a dictionary to represent the DP table
        # The key is a tuple (first_char, last_char), and the value is the length
        dp = {}
        dp[(first_char, last_char)] = length
        for i in range(1, n):
            current_word = words[i]
            current_first = current_word[0]
            current_last = current_word[-1]
            current_length = len(current_word)
            new_dp = {}
            for (prev_first, prev_last), prev_length in dp.items():
                # Option 1: join previous string with current word
                if prev_last == current_first:
                    new_length = prev_length + current_length - 1
                else:
                    new_length = prev_length + current_length
                new_first = prev_first
                new_last = current_last
                key = (new_first, new_last)
                if key in new_dp:
                    if new_length < new_dp[key]:
                        new_dp[key] = new_length
                else:
                    new_dp[key] = new_length
                # Option 2: join current word with previous string
                if current_last == prev_first:
                    new_length = current_length + prev_length - 1
                else:
                    new_length = current_length + prev_length
                new_first = current_first
                new_last = prev_last
                key = (new_first, new_last)
                if key in new_dp:
                    if new_length < new_dp[key]:
                        new_dp[key] = new_length
                else:
                    new_dp[key] = new_length
            dp = new_dp
        # Find the minimum length in the final DP table
        min_length = float('inf')
        for key, length in dp.items():
            if length < min_length:
                min_length = length
        return min_length