class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Initialize the count of possible strings to 1 (the word itself is always a possibility)
        count = 1

        # Initialize a variable to keep track of consecutive characters
        consecutive_count = 1

        # Iterate through the string starting from the second character
        for i in range(1, len(word)):
            # If the current character is the same as the previous one
            if word[i] == word[i - 1]:
                # Increment the consecutive count
                consecutive_count += 1
            else:
                # If the consecutive count is greater than 1, it means we have a sequence of repeated characters
                if consecutive_count > 1:
                    # Add the number of possible substrings that can be formed by removing some of the repeated characters
                    count += consecutive_count - 1
                # Reset the consecutive count for the new character
                consecutive_count = 1

        # Check for the last sequence of consecutive characters
        if consecutive_count > 1:
            count += consecutive_count - 1

        return count