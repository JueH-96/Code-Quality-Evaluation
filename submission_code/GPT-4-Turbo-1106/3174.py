class Solution:
    def minChanges(self, s: str) -> int:
        # Count the number of changes needed to make all characters in even positions the same
        # and all characters in odd positions the same.
        changes_for_01 = changes_for_10 = 0
        for i in range(0, len(s), 2):
            if s[i] == '0':
                changes_for_10 += 1
            else:
                changes_for_01 += 1
            if s[i+1] == '1':
                changes_for_10 += 1
            else:
                changes_for_01 += 1
        
        # The minimum number of changes is the minimum of the two counts.
        return min(changes_for_01, changes_for_10)