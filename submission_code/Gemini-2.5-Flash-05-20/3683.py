class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        L = len(word)

        # Calculate the maximum possible length a single string segment can have.
        # If there are `numFriends` segments, and each must be non-empty (at least 1 character),
        # then `numFriends - 1` characters are 'reserved' for the other `numFriends - 1` segments.
        # So, the current segment can take up to `L - (numFriends - 1)` characters.
        # Example: word="abcde", numFriends=3. L=5. numFriends-1 = 2.
        # Max length of one segment = 5 - 2 = 3.
        # e.g., "abc" (len 3), remaining "de" (len 2) can be split into "d", "e".
        # The constraint 1 <= numFriends <= word.length ensures max_allowed_len >= 1.
        max_allowed_len = L - (numFriends - 1)

        # Initialize the lexicographically largest string found so far.
        # An empty string is lexicographically smaller than any non-empty string.
        max_lex_string = ""

        # Iterate through all possible starting positions for a substring.
        for i in range(L):
            # For a fixed starting position `i`, the lexicographically largest
            # string among all substrings starting at `i` is the longest one.
            # This is because if string A is a prefix of string B, and A != B,
            # then A < B lexicographically (e.g., "a" < "ab").
            # The length of this longest substring is capped by `max_allowed_len`.
            # Also, it cannot extend beyond the end of the word.
            
            # Calculate the exclusive end index for the current candidate string.
            # It's either `i + max_allowed_len` (respecting the max_allowed_len constraint)
            # or the actual end of the word `L` (if `i + max_allowed_len` goes beyond L),
            # whichever comes first.
            end_idx = min(L, i + max_allowed_len)
            
            # Extract the current candidate substring.
            # Since `max_allowed_len` is guaranteed to be at least 1,
            # `end_idx - i` will also be at least 1, ensuring non-empty substrings.
            current_candidate_string = word[i:end_idx]

            # Compare and update the lexicographically largest string.
            if current_candidate_string > max_lex_string:
                max_lex_string = current_candidate_string
        
        return max_lex_string