from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)

        # Initialize dp arrays
        dp_prefix = [0] * n
        dp_suffix = [0] * n

        # Fill dp_prefix array
        dp_prefix[0] = len(words[0])
        for i in range(1, n):
            dp_prefix[i] = dp_prefix[i - 1] + len(words[i])
            if words[i - 1][-1] == words[i][0]:
                dp_prefix[i] -= 1

        # Fill dp_suffix array
        dp_suffix[0] = len(words[0])
        for i in range(1, n):
            dp_suffix[i] = dp_suffix[i - 1] + len(words[i])
            if words[i][-1] == words[i - 1][0]:
                dp_suffix[i] -= 1

        # Calculate the minimum length
        min_length = min(dp_prefix[-1], dp_suffix[-1])

        # Check for possible optimizations by joining from both ends
        for i in range(1, n):
            combined_length = dp_prefix[i - 1] + dp_suffix[n - 1 - i]
            if words[i - 1][-1] == words[n - 1 - i][0]:
                combined_length -= 1
            min_length = min(min_length, combined_length)

        return min_length