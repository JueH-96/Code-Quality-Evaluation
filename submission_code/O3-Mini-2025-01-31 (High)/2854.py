from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # dp maps (first_char, last_char) -> minimum accumulated length so far.
        # Initially, we use the very first word.
        first_word = words[0]
        dp = {(first_word[0], first_word[-1]): len(first_word)}
        
        # Process each subsequent word.
        for i in range(1, len(words)):
            word = words[i]
            next_dp = {}
            word_len = len(word)
            for (first, last), curr_len in dp.items():
                # Option 1: join(previous_string, word)
                # The new string starts with 'first' (the old first char)
                # and ends with word[-1].
                # We reduce the length by one if the previous last equals word[0].
                reduction = 1 if last == word[0] else 0
                new_len = curr_len + word_len - reduction
                new_state = (first, word[-1])
                if new_state not in next_dp or new_len < next_dp[new_state]:
                    next_dp[new_state] = new_len
                
                # Option 2: join(word, previous_string)
                # Now the new string starts with word[0] and ends with the old last.
                # Here we reduce length by one if word[-1] equals the previous first.
                reduction = 1 if word[-1] == first else 0
                new_len = curr_len + word_len - reduction
                new_state = (word[0], last)
                if new_state not in next_dp or new_len < next_dp[new_state]:
                    next_dp[new_state] = new_len
                
            dp = next_dp
        
        # The answer is the smallest accumulated length among all states.
        return min(dp.values())

# For testing
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimizeConcatenatedLength(["aa", "ab", "bc"]))  # Expected output: 4
    print(sol.minimizeConcatenatedLength(["ab", "b"]))          # Expected output: 2
    print(sol.minimizeConcatenatedLength(["aaa", "c", "aba"]))   # Expected output: 6