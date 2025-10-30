import collections

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        """
        Solves the problem by identifying the characters that survive until the last operation.

        The key insight is that the number of operations is determined by the character
        with the highest frequency. Let this be max_freq. After `max_freq - 1` operations,
        only the `max_freq`-th occurrences of the characters that appear `max_freq` times
        will remain. These characters, in their original relative order, form the
        string just before the last operation.

        The algorithm proceeds as follows:
        1. Calculate the frequency of each character in the string `s`.
        2. Determine the maximum frequency, `max_freq`.
        3. The characters that will form the final non-empty string are those whose
           total frequency is `max_freq`.
        4. To construct the result, we iterate through the original string `s` and keep
           track of the counts of characters seen so far.
        5. When the count of a character reaches `max_freq`, it means we have found
           its last occurrence. We append this character to our result.
        6. Iterating through `s` from left to right ensures that the relative order
           of these last occurrences is preserved.
        """
        
        # Step 1: Calculate character frequencies.
        counts = collections.Counter(s)
        
        # Step 2: Find the maximum frequency.
        # Since s.length >= 1, counts will not be empty, so max() is safe.
        max_freq = max(counts.values())
        
        # Step 3 & 4: Build the result string.
        result_chars = []
        # This dictionary tracks the number of times we have seen each character so far
        # during our traversal of `s`.
        current_counts = collections.defaultdict(int)
        
        for char in s:
            current_counts[char] += 1
            # A character is part of the final non-empty string if we are at its
            # `max_freq`-th occurrence. This is only possible for characters that appear
            # exactly `max_freq` times in total. By adding them to the result list
            # as we encounter them in the original string, we preserve their relative order.
            if current_counts[char] == max_freq:
                result_chars.append(char)
                
        return "".join(result_chars)