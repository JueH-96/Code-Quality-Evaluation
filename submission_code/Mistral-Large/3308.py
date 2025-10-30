class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Create a list to keep track of the last occurrence of each character
        last_occurrence = [-1] * 26

        # Iterate over the string to fill the last_occurrence list
        for i, char in enumerate(s):
            last_occurrence[ord(char) - ord('a')] = i

        # Find the maximum index where the last occurrence happens
        max_index = max(last_occurrence)

        # The second last operation will leave the string up to the max_index - 1
        return s[:max_index]