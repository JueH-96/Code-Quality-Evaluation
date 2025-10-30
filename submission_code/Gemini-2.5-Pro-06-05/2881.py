class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        """
        Splits each string in a list by a separator and returns a flattened list of non-empty strings.
        """
        
        # A nested list comprehension provides a concise and Pythonic solution.
        # This is a common pattern for flattening a list of lists while filtering.
        
        # The expression can be read from left to right:
        # 1. `for word in words`: Iterate through each string in the input list.
        # 2. `for part in word.split(separator)`: For each word, split it into parts
        #    and iterate through those parts. This effectively flattens the structure.
        # 3. `if part`: Filter out any parts that are empty strings. An empty string
        #    evaluates to False in a boolean context, making this a concise check.
        # 4. `part`: The expression for the items to be included in the new list,
        #    which are the non-empty parts themselves.
        
        return [
            part
            for word in words
            for part in word.split(separator)
            if part
        ]