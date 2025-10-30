class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert the string to a list to allow modifications
        s_list = list(s)

        # Iterate through the string to find the first position where a swap can be made
        for i in range(len(s_list) - 1):
            # Check if the current digit is greater than the next digit and they have the same parity
            if s_list[i] > s_list[i + 1] and (int(s_list[i]) % 2 == int(s_list[i + 1]) % 2):
                # Swap the digits
                s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                break

        # Convert the list back to a string
        return ''.join(s_list)