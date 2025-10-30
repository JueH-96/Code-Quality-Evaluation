class Solution:
    def possibleStringCount(self, word: str) -> int:
        # The idea is to observe that the final printed word is formed by consecutive groups
        # of the same letter. For each group:
        #   • If there was no error, then the intended group is exactly as printed.
        #   • If there was an error (i.e. Alice held the key too long), then the intended group
        #     is a proper prefix of the group (at least one character, but strictly fewer than printed).
        # Since the error may have occurred at most once (i.e. at most one group could be "trimmed"),
        # the total number of possible original strings is:
        #
        #     (possibility with no error) + (for each group that can be trimmed, the number of ways to trim)
        #
        # In every group without error we have 1 possibility (the group is as printed).
        # For a group with count L (L>1), if an error occurred on that group, there are (L-1) possibilities
        # (original group length could be any of 1, 2, ..., L-1).
        #
        # Hence the answer is:
        #       1 (all groups typed correctly) + Sum[(L - 1)] for every group with L > 1.
        
        total = 1  # possibility if no error occurred in any group.
        n = len(word)
        i = 0
        
        while i < n:
            j = i
            while j < n and word[j] == word[i]:
                j += 1
            group_len = j - i
            if group_len > 1:
                total += (group_len - 1)
            i = j
            
        return total

# For testing purposes:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.possibleStringCount("abbcccc"))  # Expected output: 5
    # Example 2:
    print(sol.possibleStringCount("abcd"))     # Expected output: 1
    # Example 3:
    print(sol.possibleStringCount("aaaa"))     # Expected output: 4