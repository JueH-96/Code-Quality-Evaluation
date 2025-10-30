class Solution:
    def maxOperations(self, s: str) -> int:
        # Convert the string to a list for easier manipulation
        s = list(s)
        n = len(s)
        operations = 0

        # Iterate over the string
        i = 0
        while i < n - 1:
            # If we find '1' followed by '0', we can perform the operation
            if s[i] == '1' and s[i + 1] == '0':
                operations += 1
                # Move the '1' to the right until it reaches the end or another '1'
                j = i + 1
                while j < n and s[j] == '0':
                    j += 1
                # Swap the '1' with the '0's in between
                for k in range(j - 1, i, -1):
                    s[k], s[k - 1] = s[k - 1], s[k]
                i = j - 1  # Continue from the new position of '1'
            else:
                i += 1

        return operations