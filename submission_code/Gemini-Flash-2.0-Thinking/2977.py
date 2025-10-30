class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        acronym = ""
        for word in words:
            if word:
                acronym += word[0]
            else:
                # If a word is empty, the acronym cannot be formed as expected.
                # However, the constraints guarantee that each word has at least length 1.
                pass
        return acronym == s