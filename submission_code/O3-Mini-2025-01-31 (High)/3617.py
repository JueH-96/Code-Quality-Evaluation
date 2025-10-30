class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Explanation:
        # When Alice types a letter, normally she gets exactly one occurrence in the output.
        # However, at most once she might “over‐press” a key which causes that single key press
        # to contribute more than one letter (i.e. at least 2 copies).
        #
        # The final output word is made up of one or more contiguous groups of the same letter.
        # Suppose one group in the final word consists of k consecutive identical characters.
        # In normal (error‐free) typing, one key press produces exactly one letter so that group
        # should have k occurrences if every key press was normal.
        #
        # If an error happened on that key press then the letter was repeated extra times.
        # In that case, the intended key press would have contributed only one letter even though
        # the output is k copies. In other words, the original intended key press count “drops”
        # below k. More precisely, if an error occurred on that group then the intended string
        # included fewer than k copies of that same letter – it could be any number m such that 1 <= m <= k-1.
        #
        # Since Alice is clumsy at most once, among all groups at most one group might have been
        # produced with an error. For each group we have two options:
        #   1. Use no error for that group: then the intended letter count equals the group’s length.
        #   2. (Only once overall) Use the error here (only if group length k >= 2): then the intended
        #      letter count can be any value from 1 up to k-1 (giving k-1 possibilities).
        #
        # Importantly, regardless of the group or how many letters it contains, the error (if any)
        # can only happen in one group. This means the total number of distinct intended strings is:
        #
        #   Total possibilities = (no error anywhere) + (sum over groups of error possibilities)
        #
        # For a group of k letters:
        #   - No-error yields 1 possibility.
        #   - Error (if k >= 2) yields (k - 1) possibilities.
        #
        # Since the no-error option is common to every group (and must occur together over all groups)
        # there is exactly 1 intended string if no error happened.
        # And if an error happened, it must be on exactly one group.
        #
        # Thus, overall,
        #   answer = 1 + sum_{group with k >= 2} (k - 1)
        
        groups = []
        n = len(word)
        # Build groups of consecutive identical letters.
        i = 0
        while i < n:
            j = i
            while j < n and word[j] == word[i]:
                j += 1
            groups.append(j - i)
            i = j

        total = 1  # no error anywhere
        # For each group that could have an error
        for k in groups:
            if k >= 2:
                total += (k - 1)
        return total


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test Examples:
    print(sol.possibleStringCount("abbcccc"))  # Expected output: 5
    print(sol.possibleStringCount("abcd"))     # Expected output: 1
    print(sol.possibleStringCount("aaaa"))     # Expected output: 4