class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Special-case: if k == 1 every number is divisible by 1.
        # Thus, the maximum n‑digit palindrome is just n copies of "9".
        if k == 1:
            return "9" * n

        # The n-digit palindrome is completely determined by its first half.
        # Let m be the “prefix length”. For even n, m = n//2;
        # for odd n, m = (n+1)//2 (because the middle digit is chosen only once).
        if n % 2 == 0:
            m = n // 2
        else:
            m = (n + 1) // 2

        # In order to test divisibility by k we work mod k.
        # Write the palindrome (digits d_0,...,d_{m-1}) in “mirror form.”
        # For even n, the palindrome is: 
        #    d_0 d_1 ... d_{m-1} | d_{m-1} ... d_1 d_0
        # and its value is:
        #    X = d_0*(10^(n-1)+10^(0)) + d_1*(10^(n-2)+10^(1)) + ... + d_{m-1}*(10^(n-m)+10^(m-1)).
        # For odd n (n = 2*m-1), the left part is d_0 ... d_{m-2} and the middle d_{m-1} appears only once:
        #    X = Σ{ i=0 to m-2 } d_i*(10^(n-1-i) + 10^(i)) + d_{m-1} * 10^(m-1).
        # We will “precompute” the coefficients (mod k) for each digit.
        # We first compute powers p[i] = 10^i mod k (for i = 0 to n-1).
        p = [1] * n
        for i in range(1, n):
            p[i] = (p[i - 1] * 10) % k

        # Build the list of coefficients c[0..m-1].
        c = [0] * m
        if n % 2 == 0:
            # Even case: c[i] = (10^(n-1-i) + 10^(i)) mod k.
            for i in range(m):
                c[i] = (p[n - 1 - i] + p[i]) % k
        else:
            # Odd case: for positions 0..m-2 the same double–appearance;
            for i in range(m - 1):
                c[i] = (p[n - 1 - i] + p[i]) % k
            # The middle digit (position m-1) appears only once.
            c[m - 1] = p[m - 1] % k  # (Remember: when n=2*m-1, 10^(n-m) = 10^(m-1).)

        # We'll use dynamic programming to decide whether we can
        # choose prefix digits so that the complete palindrome is divisible by k.
        # Define dp[i][r] for 0 <= i <= m and 0 <= r < k:
        #   dp[i][r] is True if for positions i ... m-1 we can choose digits (with the proper restrictions)
        #   so that starting with current modulus r, we end with a total sum ≡ 0 (mod k).
        # Our recurrence is:
        #   For position i, if we choose digit d then the new remainder is: (r + d*c[i]) mod k.
        # Base: dp[m][r] is True only if r == 0.
        dp = [[False] * k for _ in range(m + 1)]
        for r in range(k):
            dp[m][r] = (r == 0)
            
        # Fill dp backwards from position m-1 to 0.
        for i in range(m - 1, -1, -1):
            # Determine the allowed digits.
            # Position 0 cannot be 0 (to avoid leading zeros).
            allowed = range(1, 10) if i == 0 else range(0, 10)
            cur_c = c[i]
            for r in range(k):
                possible = False
                for d in allowed:
                    new_r = (r + d * cur_c) % k
                    if dp[i + 1][new_r]:
                        possible = True
                        break
                dp[i][r] = possible

        # Reconstruct the lexicographically maximum prefix.
        # We start with remainder r = 0 at position 0.
        if not dp[0][0]:
            # In principle a solution should exist.
            return ""
        prefix_digits = []
        r = 0
        for i in range(m):
            # At each position choose the largest allowed digit that leads
            # to at least one valid completion.
            allowed = range(9, 0, -1) if i == 0 else range(9, -1, -1)
            cur_c = c[i]
            chosen = None
            for d in allowed:
                new_r = (r + d * cur_c) % k
                if dp[i + 1][new_r]:
                    chosen = d
                    r = new_r
                    prefix_digits.append(str(d))
                    break
            # The dp guarantees that one digit will be chosen.
            if chosen is None:
                return ""
        prefix = "".join(prefix_digits)

        # Construct the full palindrome.
        # For even n, mirror the entire prefix;
        # for odd n, mirror all but the middle digit.
        if n % 2 == 0:
            return prefix + prefix[::-1]
        else:
            return prefix + prefix[:-1][::-1]

# --- Testing code: ---
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.largestPalindrome(3, 5))  # Expected output: "595"
    # Example 2:
    print(sol.largestPalindrome(1, 4))  # Expected output: "8"
    # Example 3:
    print(sol.largestPalindrome(5, 6))  # Expected output: "89898"