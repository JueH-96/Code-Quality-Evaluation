class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        last = [None] * 26  # To store the last occurrence index of each character
        for i, ch in enumerate(s):
            last[ord(ch) - ord('a')] = i  # Update the last occurrence index of the character
        max_idx = max(filter(lambda x: x is not None, last))  # Find the index of the last occurrence
        result = []  # To store characters in the result string
        last_taken = set()  # To keep track of characters removed

        for i in range(max_idx, -1, -1):  # Iterate from the index of the last occurrence to the start
            ch = s[i]
            ch_index = ord(ch) - ord('a')
            if last[ch_index] is not None and last[ch_index] not in last_taken:  # Check if the character can be added to the result
                result.append(ch)
                last_taken.add(last[ch_index])  # Mark the character as taken
                if len(result) == 26:  # Stop when result includes all characters
                    break

        return ''.join(reversed(result[:25]))  # Reverse the result and take the first 25 characters