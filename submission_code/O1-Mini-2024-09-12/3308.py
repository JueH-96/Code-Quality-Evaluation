class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict

        # Dictionary to keep count of occurrences for each character
        count = defaultdict(int)
        # List to store operation number for each character
        operations = []
        max_op = 0

        for c in s:
            count[c] += 1
            operations.append(count[c])
            if count[c] > max_op:
                max_op = count[c]

        # Collect characters that are removed in the last operation
        result = []
        for c, op in zip(s, operations):
            if op == max_op:
                result.append(c)

        return ''.join(result)