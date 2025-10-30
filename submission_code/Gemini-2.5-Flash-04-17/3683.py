class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        # The maximum possible length for any individual string piece
        # that can result from splitting 'word' into 'numFriends' non-empty parts.
        # If a piece has length `L`, the remaining `n - L` characters must be
        # partitionable into `numFriends - 1` non-empty pieces.
        # Minimum length for numFriends - 1 non-empty pieces is numFriends - 1.
        # So, `n - L >= numFriends - 1`, which implies `L <= n - numFriends + 1`.
        max_allowed_len = n - numFriends + 1

        # We need to find the lexicographically largest substring of 'word'
        # that has a length `L` such that `1 <= L <= max_allowed_len`.

        # Initialize the best substring found so far.
        # A valid candidate substring must start at some index i and have length at least 1.
        # The substring starting at index 0 with the maximum allowed length
        # from that position is a valid candidate. Its length is `min(n, max_allowed_len)`.
        # Since `1 <= numFriends <= n`, we have `1 <= max_allowed_len <= n`.
        # Thus, `min(n, max_allowed_len)` is at least 1, so word[0 : min(n, max_allowed_len)]
        # is a non-empty valid substring.
        best_substring = word[0 : min(n, max_allowed_len)]

        # Iterate through all possible starting indices for substrings, from 1 up to n-1.
        for i in range(1, n):
            # Determine the maximum possible length for a substring starting at index i.
            # This length is limited by the remaining characters in the string (n - i)
            # and the overall maximum allowed length for any piece (max_allowed_len).
            current_possible_len = min(n - i, max_allowed_len)

            # For a fixed starting index `i`, the lexicographically largest substring
            # among all substrings starting at `i` with lengths from 1 up to
            # `current_possible_len` is the longest one: `word[i : i + current_possible_len]`.
            # This is because if string A is a prefix of string B (A != B), then B is
            # lexicographically greater than A.
            current_substring = word[i : i + current_possible_len]

            # Compare the current largest substring (starting at index i) with the
            # overall best substring found across all previous starting indices.
            # Update the overall best if the current one is lexicographically greater.
            if current_substring > best_substring:
                best_substring = current_substring

        # After checking all possible starting indices, best_substring holds the
        # lexicographically largest string among all candidates.
        return best_substring