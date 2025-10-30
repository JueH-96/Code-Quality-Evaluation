class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        # Initialize counters for changes needed to make the string all '0's or all '1's
        changes_to_zero = 0
        changes_to_one = 0

        # Traverse the string and count the changes needed
        for i in range(n):
            if (i % 2 == 0 and s[i] == '1') or (i % 2 == 1 and s[i] == '0'):
                changes_to_zero += 1
            if (i % 2 == 0 and s[i] == '0') or (i % 2 == 1 and s[i] == '1'):
                changes_to_one += 1

        # Return the minimum changes needed
        return min(changes_to_zero, changes_to_one)