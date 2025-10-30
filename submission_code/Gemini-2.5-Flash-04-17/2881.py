from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        """
        Splits each string in words by the separator and returns a list of 
        non-empty resulting strings.

        Args:
            words: A list of strings.
            separator: A character used to split the strings.

        Returns:
            A list of strings formed after splitting and filtering out empty strings,
            maintaining the original order.
        """
        result = []
        for word in words:
            # Split the current word by the separator.
            # The split method handles consecutive separators and
            # leading/trailing separators by producing empty strings.
            parts = word.split(separator)
            # Iterate through the resulting parts.
            for part in parts:
                # If the part is not an empty string, add it to the result list.
                if part != "":
                    result.append(part)
        return result