class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)

        # If no operations are allowed, return the length of the longest substring of identical characters
        if numOps == 0:
            max_len = 1
            current_len = 1
            for i in range(1, n):
                if s[i] == s[i - 1]:
                    current_len += 1
                else:
                    max_len = max(max_len, current_len)
                    current_len = 1
            return max(max_len, current_len)

        # Initialize the minimum length to the length of the string
        min_length = n

        # Try flipping characters from the start and end of the string
        for i in range(n):
            if i > numOps:
                break
            left_flips = i
            right_flips = numOps - left_flips
            left = i
            right = n - 1 - right_flips

            if left > right:
                break

            # Find the longest substring of identical characters in the modified string
            current_length = 1
            max_length = 1

            for j in range(left, right):
                if s[j] == s[j + 1]:
                    current_length += 1
                else:
                    max_length = max(max_length, current_length)
                    current_length = 1

            max_length = max(max_length, current_length)
            min_length = min(min_length, max_length)

        return min_length