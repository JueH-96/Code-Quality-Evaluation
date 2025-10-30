class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        for i in range(0, n, 2):
            first_char = s[i]
            second_char = s[i+1]
            cost_to_00 = 0
            if first_char == '1':
                cost_to_00 += 1
            if second_char == '1':
                cost_to_00 += 1
            cost_to_11 = 0
            if first_char == '0':
                cost_to_11 += 1
            if second_char == '0':
                cost_to_11 += 1
            changes += min(cost_to_00, cost_to_11)
        return changes