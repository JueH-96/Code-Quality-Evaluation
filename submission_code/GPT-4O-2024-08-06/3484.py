class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert string to list for easier manipulation
        s_list = list(s)
        n = len(s_list)

        # Iterate over the string
        for i in range(n - 1):
            # Check if adjacent digits have the same parity
            if (int(s_list[i]) % 2 == int(s_list[i + 1]) % 2):
                # If they do, check if swapping them makes the string smaller
                if s_list[i] > s_list[i + 1]:
                    # Swap the digits
                    s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                    # Since we can only swap once, we break after the first swap
                    break

        # Convert list back to string and return
        return ''.join(s_list)