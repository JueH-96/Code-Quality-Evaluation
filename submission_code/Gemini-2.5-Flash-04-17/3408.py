class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        seen_lower = set()
        seen_upper = set()

        # Populate sets with seen lowercase and uppercase characters
        for char in word:
            if char.islower():
                seen_lower.add(char)
            elif char.isupper():
                seen_upper.add(char)

        special_count = 0
        # Check each seen lowercase character if its uppercase counterpart was also seen
        # Iterating through seen_lower ensures we only check characters
        # that actually appeared in lowercase in the word.
        for char in seen_lower:
            # For each lowercase character, check if its uppercase version is in the set of seen uppercase characters
            if char.upper() in seen_upper:
                special_count += 1

        return special_count