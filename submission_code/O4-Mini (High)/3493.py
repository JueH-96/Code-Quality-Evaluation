class Solution:
    def maxOperations(self, s: str) -> int:
        """
        Each time we have a '1' followed by a '0' in the string,
        we can perform one operation for that '1'.  Moreover,
        the cost of that operation is equal to how many '1's
        are to the left of (and including) that position.
        Summing over all i with s[i]=='1' and s[i+1]=='0' gives
        the maximum number of operations.
        """
        ans = 0
        ones_count = 0
        n = len(s)
        # We only look up to n-2, since we check s[i+1].
        for i in range(n - 1):
            if s[i] == '1':
                # Count this '1'
                ones_count += 1
                # If it's followed by '0', we can apply an operation
                ans += ones_count if s[i + 1] == '0' else 0
        return ans