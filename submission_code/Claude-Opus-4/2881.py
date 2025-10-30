class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by separator
            parts = word.split(separator)
            # Add non-empty parts to result
            for part in parts:
                if part:  # Only add non-empty strings
                    result.append(part)
        return result