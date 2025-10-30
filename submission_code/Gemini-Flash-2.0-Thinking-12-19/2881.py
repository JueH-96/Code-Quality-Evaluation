class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            split_parts = word.split(separator)
            for part in split_parts:
                if part != "":
                    result.append(part)
        return result