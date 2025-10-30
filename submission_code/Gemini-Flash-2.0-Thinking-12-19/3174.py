class Solution:
    def minChanges(self, s: str) -> int:
        changes = 0
        n = len(s)
        for i in range(0, n, 2):
            first_char = s[i]
            second_char = s[i+1]
            ones_count = 0
            zeros_count = 0
            if first_char == '1':
                ones_count += 1
            else:
                zeros_count += 1
            if second_char == '1':
                ones_count += 1
            else:
                zeros_count += 1
            changes += min(ones_count, zeros_count)
        return changes