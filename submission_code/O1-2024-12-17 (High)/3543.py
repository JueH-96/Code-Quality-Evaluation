class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        # We will iterate over all starting points
        for i in range(n):
            zero_count = 0
            one_count = 0
            # Expand the current substring one character at a time
            for j in range(i, n):
                if s[j] == '0':
                    zero_count += 1
                else:
                    one_count += 1
                # Check if it satisfies the k-constraint
                if zero_count <= k or one_count <= k:
                    result += 1
        return result