class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        acronym = ""
        for word in words:
            if not word:
                return False
            acronym += word[0]
        return acronym == s