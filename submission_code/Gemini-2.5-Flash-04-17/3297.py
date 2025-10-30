class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        # We are looking for the minimum time t > 0 when the string can revert to its initial state.
        # At time t, after t operations, the first t*k characters of the original string have been removed
        # from the prefix, and t*k arbitrary characters have been added to the end.
        # The string at time t starts with the segment that was originally at index t*k,
        # which is word[t*k:] (if t*k < n). This segment has length max(0, n - t*k).
        # For the string at time t to be equal to the initial word, the prefix of length n must be word.
        # The first max(0, n - t*k) characters of the string at time t are fixed from the original word
        # and are exactly word[t*k:].
        # These characters must match the first max(0, n - t*k) characters of the original word,
        # which is word[:max(0, n - t*k)] which simplifies to word[:n - t*k] since slicing handles <=0 lengths.
        # The remaining characters in the string at time t are arbitrary, and we can choose them
        # to match the suffix word[n - t*k :].
        # Therefore, the string can revert at time t if and only if the fixed prefix matches the required prefix
        # of the original string, i.e., word[t*k:] == word[:n - t*k].

        # We check this condition for t = 1, 2, 3, ...
        t = 0
        while True:
            t += 1
            
            # Calculate the total number of characters conceptually removed from the prefix
            # of the original word after t seconds.
            chars_removed = t * k
            
            # Check if the suffix of the original word starting at index chars_removed
            # is equal to the prefix of the original word of length n - chars_removed.
            # Python slicing correctly handles edge cases where `chars_removed >= n` or `n - chars_removed <= 0`,
            # returning empty strings. The comparison "" == "" is true in these cases.
            
            suffix = word[chars_removed:]
            prefix = word[:n - chars_removed]
            
            if suffix == prefix:
                # This is the smallest t > 0 for which the condition holds.
                return t

            # The loop is guaranteed to find a solution because when `chars_removed >= n`,
            # the condition `word[chars_removed:] == word[:n - chars_removed]` simplifies to `"" == ""`,
            # which is always true. The smallest t > 0 for which `chars_removed >= n` is ceil(n / k).
            # Thus, the loop will terminate at or before this time.
            # Constraints n <= 50, k >= 1 ensure ceil(n/k) <= 50, making the loop efficient.