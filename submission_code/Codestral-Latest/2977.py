class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # Initialize an empty string to build the acronym
        acronym = ""

        # Iterate through each word in the list
        for word in words:
            # Append the first character of the word to the acronym
            acronym += word[0]

        # Check if the built acronym matches the given string s
        return acronym == s