class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Helper function to check if a substring is a semi-palindrome
        def is_semi_palindrome(start, end):
            for i in range(start, end, 2):
                if s[i] != s[end - (i - start)]:
                    return False
            return True
        
        # Helper function to count the number of changes required to make a substring a semi-palindrome
        def count_changes(start, end):
            changes = 0
            for i in range(start, end):
                if not is_semi_palindrome(start, end):
                    changes += 1
                    start += 1
            return changes
        
        # Initialize the minimum number of changes required
        min_changes = float('inf')
        
        # Try all possible partitions of the string into k substrings
        for i in range(1, n // k + 1):
            changes = 0
            for j in range(i, n, i):
                changes += count_changes(j - i, j)
            min_changes = min(min_changes, changes)
        
        return min_changes