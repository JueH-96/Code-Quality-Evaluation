class Solution:
    def getSmallestString(self, s: str) -> str:
        # Convert the string to a list of characters for easy swapping
        s_list = list(s)
        
        # Iterate through the string to find a pair of adjacent digits with the same parity
        for i in range(len(s) - 1):
            # Check if the current digit and the next one have the same parity
            if int(s_list[i]) % 2 == int(s_list[i + 1]) % 2:
                # Check if swapping them will result in a lexicographically smaller string
                if s_list[i] > s_list[i + 1]:
                    # Swap the digits
                    s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                    # Since we can swap only once, break the loop after the first swap
                    break
        
        # Convert the list of characters back to a string and return it
        return ''.join(s_list)