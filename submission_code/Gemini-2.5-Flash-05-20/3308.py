import collections

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Step 1: Calculate the initial frequencies of all characters in the string s.
        # This helps us determine the total number of operations (rounds).
        char_frequencies = collections.Counter(s)

        # Step 2: Determine the maximum frequency among all characters.
        # This maximum frequency, let's call it `max_freq`, is the total number of operations
        # that will be performed until the string becomes empty.
        # Example: If 'a' appears 5 times, 'b' appears 3 times, 'c' 1 time,
        # 'a' will require 5 rounds to be fully removed, 'b' 3 rounds, 'c' 1 round.
        # The process stops when ALL characters are gone, so it runs for `max_freq` rounds.
        # Constraints guarantee 1 <= s.length, so char_frequencies will not be empty.
        max_freq = max(char_frequencies.values())
        
        # The problem asks for the string just *before* the last operation.
        # This means we need the state of the string at the beginning of the `max_freq`-th operation.

        # Key Insight: The `i`-th occurrence of any character (relative to its position in the original string)
        # is removed during the `i`-th operation.
        # Therefore, the string remaining just before the `max_freq`-th (final) operation will consist
        # of precisely those characters that are the `max_freq`-th occurrence of their type in the
        # *original* string `s`. Characters with fewer than `max_freq` occurrences would have been
        # entirely removed in earlier operations. Characters with more than `max_freq` occurrences
        # are not possible, as `max_freq` is the highest count.

        # Step 3: Iterate through the original string `s` to identify and collect
        # the characters that are the `max_freq`-th occurrence of their type.
        # We need to maintain their original relative order.
        
        # `current_counts` will keep track of how many times each character has appeared
        # so far as we traverse the string `s` from left to right.
        current_counts = collections.defaultdict(int)
        
        # `result_chars` will store the characters that form our final answer.
        result_chars = []

        for char_code in s:
            current_counts[char_code] += 1
            # If this character is the `max_freq`-th instance of its type (in the original string order),
            # it means it is one of the characters that will be present in the string
            # right before the very last operation.
            if current_counts[char_code] == max_freq:
                result_chars.append(char_code)
        
        # Step 4: Join the collected characters to form the final string.
        return "".join(result_chars)