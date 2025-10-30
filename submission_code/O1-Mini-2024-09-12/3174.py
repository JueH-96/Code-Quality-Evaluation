class Solution:
    def minChanges(self, s: str) -> int:
        total_changes = 0
        for i in range(0, len(s), 2):
            block = s[i:i+2]
            count_zero = block.count('0')
            changes_to_zero = count_zero
            changes_to_one = 2 - count_zero
            total_changes += min(changes_to_zero, changes_to_one)
        return total_changes