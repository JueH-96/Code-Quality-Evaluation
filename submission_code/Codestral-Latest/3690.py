class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)

        # Helper function to count the length of the longest substring of identical characters
        def longest_substring_length(s):
            max_len = 1
            current_len = 1
            for i in range(1, n):
                if s[i] == s[i - 1]:
                    current_len += 1
                else:
                    max_len = max(max_len, current_len)
                    current_len = 1
            max_len = max(max_len, current_len)
            return max_len

        # If numOps is 0, return the length of the longest substring of identical characters
        if numOps == 0:
            return longest_substring_length(s)

        # Try flipping each bit and see the effect on the longest substring length
        min_length = n
        for i in range(n):
            # Flip the bit at index i
            new_s = s[:i] + ('1' if s[i] == '0' else '0') + s[i+1:]
            # Calculate the length of the longest substring of identical characters after flipping
            min_length = min(min_length, longest_substring_length(new_s))

        return min_length