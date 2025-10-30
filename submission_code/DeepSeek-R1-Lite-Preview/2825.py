class Solution:
    def minimizedStringLength(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        # Initialize for substrings of length 1
        for i in range(n):
            dp[i][i] = 1
        
        # Fill the dp table for substrings of length 2 to n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = j - i + 1  # Maximum possible length
                for k in range(i, j + 1):
                    # Find closest duplicate to the left of k
                    left = -1
                    for l in range(k - 1, i - 1, -1):
                        if s[l] == s[k]:
                            left = l
                            break
                    # Find closest duplicate to the right of k
                    right = -1
                    for m in range(k + 1, j + 1):
                        if s[m] == s[k]:
                            right = m
                            break
                    # Calculate the new length after deletion
                    if left == -1 and right == -1:
                        new_length = dp[i][j] - 0  # No deletion
                    elif left != -1 and right != -1:
                        if left < i or right > j:
                            new_length = dp[i][j] - 2
                        else:
                            new_length = (dp[i][left - 1] if left - 1 >= i else 0) + \
                                         (dp[left + 1][right - 1] if right - 1 >= left + 1 else 0) + \
                                         (dp[right + 1][j] if right + 1 <= j else 0)
                    elif left != -1:
                        new_length = dp[i][j] - 1
                    elif right != -1:
                        new_length = dp[i][j] - 1
                    dp[i][j] = min(dp[i][j], new_length)
        
        return dp[0][n - 1]