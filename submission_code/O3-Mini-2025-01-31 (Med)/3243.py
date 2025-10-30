class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Explanation:
        # Every valid x must end with s and have all digits ≤ limit.
        # Write x = m * (10^(len(s))) + int(s). Because s is a fixed suffix and its digits are already ≤ limit,
        # it remains to choose m (≥ 0) such that:
        #   start ≤ m * B + s_val ≤ finish              where B = 10^(len(s)) and s_val = int(s).
        # We also require that the “prefix number” m (when written in its natural decimal representation
        # with no extra leading zeros) has every digit ≤ limit.
        #
        # So find the allowed m range:
        #   m_min = 0 if s_val >= start, else ceiling((start - s_val) / B)
        #   m_max = floor((finish - s_val) / B)
        #
        # Then we simply count how many m in [m_min, m_max] satisfy that each digit (in the normal representation)
        # is at most limit.
        #
        # To count valid m, we can use a digit DP.
        
        B = 10 ** len(s)
        s_val = int(s)
        
        # If even the smallest candidate (x = s_val) is too large, no solution:
        if finish < s_val:
            return 0
        
        # Compute m range. Note that if start <= s_val then m can be 0 (x = s_val).
        if start <= s_val:
            m_min = 0
        else:
            m_min = (start - s_val + B - 1) // B  # Ceiling division
        
        m_max = (finish - s_val) // B
        if m_max < m_min:
            return 0
        
        # dp to count numbers up to X with each digit in {0,1,...,limit}
        from functools import lru_cache
        
        def count_valid(X: int, limit: int) -> int:
            if X < 0:  # No numbers
                return 0
            xs = str(X)
            n = len(xs)
            
            # DP state: (i, tight, started)
            # i: current index in xs,
            # tight: whether we are bound by the prefix of X
            # started: whether we have already chosen a nonzero digit (so that our number's representation is started)
            @lru_cache(maxsize=None)
            def dp(i: int, tight: bool, started: bool) -> int:
                if i == n:
                    # At the end, if we have not started, that corresponds to the number 0.
                    # Both a non-started number and a started number represent a valid number.
                    return 1
                res = 0
                # Determine the current maximum digit we can choose.
                # If tight is True, we cannot exceed the digit of X at position i.
                up = int(xs[i]) if tight else limit
                # Try every digit d from 0 to up. (Note that even when not started, we allow d = 0 freely;
                # a number like 002 becomes 2 in its canonical representation — which is fine.)
                for d in range(0, up + 1):
                    if d > limit:
                        # Should not happen because up <= limit when not tight.
                        continue
                    ntight = tight and (d == up)
                    nstarted = started or (d != 0)
                    res += dp(i + 1, ntight, nstarted)
                return res
            
            return dp(0, True, False)
        
        # Count valid m in range [m_min, m_max]:
        ans = count_valid(m_max, limit)
        if m_min - 1 >= 0:
            ans -= count_valid(m_min - 1, limit)
        return ans


# The following code is provided for local testing.
if __name__ == '__main__':
    # Example test cases:
    sol = Solution()
    # Example 1
    print(sol.numberOfPowerfulInt(1, 6000, 4, "124"))   # Expected output: 5
    # Example 2
    print(sol.numberOfPowerfulInt(15, 215, 6, "10"))    # Expected output: 2
    # Example 3
    print(sol.numberOfPowerfulInt(1000, 2000, 4, "3000"))  # Expected output: 0

    # For compatibility with the problem statement's "starter code" format, one might not include input parsing.