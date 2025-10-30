class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        ans = 0
        cur_sum = 0  # Number of '1's before the current index
        i = 0
        while i < n:
            if s[i] == '0' and (i == 0 or s[i - 1] == '1'):
                # Start of a '0' block, add the number of '1's before this block
                ans += cur_sum
                # Skip the entire '0' block
                while i < n and s[i] == '0':
                    i += 1
            else:
                # If s[i] == '1', increment cur_sum
                if s[i] == '1':
                    cur_sum += 1
                # Move to the next index
                i += 1
        return ans