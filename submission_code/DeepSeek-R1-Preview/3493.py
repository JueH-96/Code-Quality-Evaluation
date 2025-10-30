class Solution:
    def maxOperations(self, s: str) -> int:
        # Split the string into groups of '1's and '0's
        groups = []
        n = len(s)
        if n == 0:
            return 0
        current = s[0]
        count = 1
        for i in range(1, n):
            if s[i] == current:
                count += 1
            else:
                groups.append((current, count))
                current = s[i]
                count = 1
        groups.append((current, count))
        
        total = 0
        # Iterate through the groups, for each '1' group followed by a '0' group, add product
        for i in range(len(groups) - 1):
            if groups[i][0] == '1' and groups[i + 1][0] == '0':
                total += groups[i][1] * groups[i + 1][1]
        
        return total