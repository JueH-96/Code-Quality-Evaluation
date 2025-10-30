class Solution:
    def minimumLength(self, s: str) -> int:
        # Convert the string to a list for easier manipulation
        s = list(s)
        i = 0

        # Iterate over the string
        while i < len(s):
            # Find the closest character to the left that is equal to s[i]
            left = -1
            for j in range(i - 1, -1, -1):
                if s[j] == s[i]:
                    left = j
                    break

            # Find the closest character to the right that is equal to s[i]
            right = -1
            for j in range(i + 1, len(s)):
                if s[j] == s[i]:
                    right = j
                    break

            # If both left and right characters are found, delete them
            if left != -1 and right != -1:
                del s[right]
                del s[left]
                # Reset i to check the new string from the beginning
                i = 0
            else:
                # Move to the next character
                i += 1

        # Return the length of the final string
        return len(s)