class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        # Initialize min_s with the original string. This is crucial because
        # the problem states "at most once swap", meaning no swap is also
        # a valid option, and might result in the smallest string.
        min_s = s 

        # Helper function to determine if a digit character represents an even number.
        # '0', '2', '4', '6', '8' are even. '1', '3', '5', '7', '9' are odd.
        def is_even(char_digit: str) -> bool:
            return int(char_digit) % 2 == 0

        # Iterate through all possible adjacent pairs (s[i], s[i+1]).
        # A swap can occur between index i and i+1.
        # So, 'i' iterates from 0 up to n-2 (inclusive).
        for i in range(n - 1):
            char1 = s[i]
            char2 = s[i+1]

            # Check if the two adjacent digits have the same parity.
            # This condition is true if both are even OR both are odd.
            if is_even(char1) == is_even(char2):
                # If they have the same parity, perform a hypothetical swap.
                # Convert the string to a list of characters for efficient modification.
                s_list = list(s)
                # Swap the characters at positions i and i+1
                s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                # Join the list back into a string to get the result of this specific swap.
                current_s_after_swap = "".join(s_list)

                # Compare the string obtained after this swap with the current minimum string.
                # If the swapped string is lexicographically smaller, update min_s.
                if current_s_after_swap < min_s:
                    min_s = current_s_after_swap
        
        # After checking all possible single valid swaps, min_s will hold
        # the lexicographically smallest string among all possibilities.
        return min_s