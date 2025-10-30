class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict, deque

        # Create a dictionary to store the positions of each character
        char_positions = defaultdict(deque)
        for idx, char in enumerate(s):
            char_positions[char].append(idx)

        # Determine the maximum number of operations for any character
        max_ops = max(len(positions) for positions in char_positions.values())

        # If max_ops is 1, the last operation will remove all characters
        if max_ops == 1:
            return s

        # Find the characters that have the maximum number of operations
        chars_with_max_ops = [char for char, positions in char_positions.items() if len(positions) == max_ops]

        # Get the last occurrence of each character with max_ops
        last_positions = []
        for char in chars_with_max_ops:
            last_positions.append(char_positions[char][-1])

        # Sort the last positions to get the order in which they appear in the string
        last_positions.sort()

        # Construct the result string by taking characters at these positions
        result = ''.join([s[pos] for pos in last_positions])

        return result