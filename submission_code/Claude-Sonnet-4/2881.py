class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        
        for word in words:
            # Split the word by separator
            split_parts = word.split(separator)
            
            # Add non-empty parts to result
            for part in split_parts:
                if part:  # Only add if not empty
                    result.append(part)
        
        return result