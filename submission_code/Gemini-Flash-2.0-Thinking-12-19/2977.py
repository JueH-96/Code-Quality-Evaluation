class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        acronym_str = ""
        for word in words:
            acronym_str += word[0]
        return acronym_str == s