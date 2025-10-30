class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        count_ones = s.count('1')
        steps = 0

        for i in range(n):
            if s[i] == '1':
                # Calculate the number of '0's to the right of the current '1'
                steps += (n - i - 1) - (count_ones - 1)
                count_ones -= 1

        return steps