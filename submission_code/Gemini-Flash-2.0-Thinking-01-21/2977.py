class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        acronym = ""
        for word in words:
            if len(word) > 0:
                acronym += word[0]
        return acronym == s