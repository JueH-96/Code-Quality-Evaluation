class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        # positions where s[i]=='1'
        ones = [i for i, ch in enumerate(s) if ch == '1']
        m = len(ones)
        if m == 0:
            return 0
        # In the final sorted state the ones occupy the rightmost m positions.
        # That is, the ideal positions for the 1's are:
        start = n - m  # first index for a '1'
        q = [start + i for i in range(m)]
        
        # dp[i] = maximum number of separate one‐step moves 
        # (i.e. operations on the i–th token) that we can “force”
        # for the i–th 1 in the optimal stretching.
        dp = [0] * m
        dp[m-1] = 1 if ones[m-1] < q[m-1] else 0  # rightmost token
        for i in range(m-2, -1, -1):
            if ones[i] >= q[i]:
                dp[i] = 0
            else:
                # if the next token is immediately adjacent,
                # then we cannot “split” further.
                if ones[i+1] == ones[i] + 1:
                    dp[i] = 1
                else:
                    # Otherwise, we can “split” at most one more time than dp[i+1],
                    # but not more than the total gap (q[i]-ones[i])
                    dp[i] = min(q[i] - ones[i], dp[i+1] + 1)
        return sum(dp)


# For testing (you can remove or comment out the testing part later):
if __name__ == '__main__':
    sol = Solution()
    # sample test cases
    tests = [
        ("1001101", 4),
        ("00111", 0),
        ("10", 1),
        ("1010", 3),
        ("111000", 3),
        ("110", 2),
        ("1011", 1),
    ]
    for inp, expected in tests:
        res = sol.maxOperations(inp)
        print(f"s = {inp!r}, expected = {expected}, got = {res}")