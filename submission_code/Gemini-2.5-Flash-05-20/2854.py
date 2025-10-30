import collections
from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)

        # dp is a dictionary where keys are (first_char, last_char) tuples
        # and values are the minimum length of the concatenated string
        # that ends with those first and last characters.
        # This approach uses space optimization by only keeping track of
        # the previous state's DP table.
        
        # Base case: For the first word (words[0]), the string is just words[0] itself.
        first_word = words[0]
        # Using collections.defaultdict(lambda: float('inf')) ensures that
        # any new (first_char, last_char) combination automatically gets an
        # initial value of infinity, allowing min() to work correctly.
        dp = collections.defaultdict(lambda: float('inf'))
        dp[(first_word[0], first_word[-1])] = len(first_word)

        # Iterate through the words starting from the second word (index 1)
        # up to the last word (index n-1).
        for i in range(1, n):
            current_word = words[i]
            current_word_len = len(current_word)
            current_word_first = current_word[0]
            current_word_last = current_word[-1]

            # next_dp will store the states for the current iteration (str_i)
            next_dp = collections.defaultdict(lambda: float('inf'))

            # Iterate over all possible (first_char, last_char) states
            # from the previous iteration (str_i-1)
            for (prev_first, prev_last), prev_len in dp.items():
                # Option 1: str_i = join(str_i-1, words[i])
                # This means str_i-1 is on the left, words[i] is on the right.
                # The length reduction occurs if the last char of str_i-1 matches
                # the first char of words[i].
                cost_reduction_1 = 1 if prev_last == current_word_first else 0
                new_len_1 = prev_len + current_word_len - cost_reduction_1
                new_first_1 = prev_first        # First char remains from str_i-1
                new_last_1 = current_word_last  # Last char comes from words[i]
                
                # Update the minimum length for this new (first_char, last_char) combination
                next_dp[(new_first_1, new_last_1)] = min(next_dp[(new_first_1, new_last_1)], new_len_1)

                # Option 2: str_i = join(words[i], str_i-1)
                # This means words[i] is on the left, str_i-1 is on the right.
                # The length reduction occurs if the last char of words[i] matches
                # the first char of str_i-1.
                cost_reduction_2 = 1 if current_word_last == prev_first else 0
                new_len_2 = current_word_len + prev_len - cost_reduction_2
                new_first_2 = current_word_first # First char comes from words[i]
                new_last_2 = prev_last           # Last char remains from str_i-1

                # Update the minimum length for this new (first_char, last_char) combination
                next_dp[(new_first_2, new_last_2)] = min(next_dp[(new_first_2, new_last_2)], new_len_2)
            
            # After processing all previous states, next_dp becomes the current dp for the next iteration.
            dp = next_dp 

        # After all words have been processed (up to words[n-1]),
        # dp contains the minimum lengths for all possible (first_char, last_char)
        # combinations for the final string str_n-1.
        # The answer is the minimum value among all these lengths.
        min_total_length = float('inf')
        for length in dp.values():
            min_total_length = min(min_total_length, length)
        
        return min_total_length