class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Each distinct character must remain at least once, and we can always
        # delete every extra copy of a character by repeatedly picking its
        # right-most occurrence as the pivot (removing one duplicate on the left
        # each time).  Hence the minimal possible length is simply the number
        # of unique characters in the string.
        return len(set(s))