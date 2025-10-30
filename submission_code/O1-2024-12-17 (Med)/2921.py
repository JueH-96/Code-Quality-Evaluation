class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        """
        We need to find how many 'stepping numbers' lie in the inclusive range [low, high].
        A stepping number is defined as one in which the absolute difference between every
        pair of adjacent digits is exactly 1. Leading zeros are not allowed, so the first
        digit must be nonzero.

        Since low and high can be up to length 100 (i.e. < 10^100), we cannot simply iterate.
        Instead, we use a combination of:
          1) Precomputation of the count of all stepping numbers of length < L.
          2) A digit-DP (dynamic programming) approach to count how many stepping numbers
             of length L are <= a given number N.

        Then, countSteppingNumbers([low, high]) = countAllUpTo(high) - countAllUpTo(low - 1).

        Steps in detail:

        1) Precompute counts of stepping numbers by length (without leading zero).
           Let preCount[L] = number of stepping numbers of exact length L.
           We do this up to L=100 with a simple recurrence:
                 dp2[L][d] = number of stepping numbers of length L ending with digit d
                            (and no leading zero).
                 dp2[1][d] = 1 if d != 0, else 0
                 dp2[L][d] = dp2[L-1][d-1] + dp2[L-1][d+1]   (if d-1 >= 0, d+1 <= 9)
           Then preCount[L] = sum(dp2[L][d] for d in [0..9]), though dp2[L][0] would be 0
           for L=1 because we can't start with zero.
           We'll keep them all mod 1e9+7.

        2) Define a function digitDP(N) that counts how many stepping numbers of length len(N)
           are <= N (again with no leading zero).
           - We'll parse N into a list of digits, digits[]. Let length = len(N).
           - We'll define a recursive DP:
               dfs(pos, last_digit, tight) returns how many valid stepping numbers from
               position pos onward, given the previous digit is 'last_digit', and 'tight'
               indicates if we are bound by the prefix of N.
               If pos == length, return 1 (we formed a valid number).
               Otherwise let limit = digits[pos] if tight==1 else 9.
               We'll iterate next_digit from 0..limit, apply the stepping constraint:
                 - If pos == 0, we skip next_digit=0 (leading zero disallowed).
                 - If pos > 0, we require abs(next_digit - last_digit)==1.
               Then we pick new_tight = tight and (next_digit == limit).
               Sum all valid transitions modulo 1e9+7.

        3) countAllUpTo(N) = sum of all stepping numbers with length < len(N) plus
           digitDP(N) (which counts valid stepping numbers of the same length as N).
           So:
               countAllUpTo(N) = sum(preCount[L] for L in [1..len(N)-1]) + digitDP(N)
           Special case: if N == "0", we define countAllUpTo("0") = 0.

        4) Finally, the answer = (countAllUpTo(high) - countAllUpTo(low - 1)) % (1e9+7),
           taking care that low - 1 might be "0" if low == "1".

        This handles all numbers up to length 100 safely and efficiently.
        """

        MOD = 10**9 + 7

        # 1) Precompute the count of stepping numbers of length L (1 <= L <= 100).
        # dp2[L][d] = number of length-L stepping numbers ending with digit d (leading digit != 0).
        max_len = 100
        dp2 = [[0]*10 for _ in range(max_len+1)]
        # Base case for length=1: digits 1..9 are valid, digit 0 is not valid for leading
        for d in range(1, 10):
            dp2[1][d] = 1

        for length in range(2, max_len+1):
            for d in range(10):
                if d-1 >= 0:
                    dp2[length][d] = (dp2[length][d] + dp2[length-1][d-1]) % MOD
                if d+1 <= 9:
                    dp2[length][d] = (dp2[length][d] + dp2[length-1][d+1]) % MOD

        # preCount[L] = total stepping numbers of length L (leading digit != 0)
        preCount = [0]*(max_len+1)
        for L in range(1, max_len+1):
            preCount[L] = sum(dp2[L][d] for d in range(10)) % MOD

        # Digit DP cache: memo[pos][last_digit][tight] => number of valid ways
        from functools import lru_cache

        def digitDP(num_str: str) -> int:
            """
            Counts how many stepping numbers of length = len(num_str)
            are <= num_str. Leading digit cannot be zero.
            """
            digits = list(map(int, num_str))
            n = len(digits)

            @lru_cache(None)
            def dfs(pos, last_digit, tight):
                # If we've placed all digits, that's one valid number
                if pos == n:
                    return 1

                limit = digits[pos] if tight else 9
                ans = 0
                for nd in range(limit + 1):
                    if pos == 0:
                        # Leading digit cannot be zero
                        if nd == 0:
                            continue
                        # No stepping constraint for the very first digit
                        new_tight = (tight and nd == limit)
                        ans = (ans + dfs(pos+1, nd, new_tight)) % MOD
                    else:
                        # Must follow stepping constraint
                        if abs(nd - last_digit) == 1:
                            new_tight = (tight and nd == limit)
                            ans = (ans + dfs(pos+1, nd, new_tight)) % MOD
                return ans

            return dfs(0, -1, True)

        def countAllUpTo(s: str) -> int:
            """
            Returns the count of all stepping numbers <= s (s may be big),
            or 0 if s == "0".
            """
            # If s == "0", no valid stepping numbers
            if s == "0":
                return 0
            L = len(s)
            # Sum all stepping numbers of length < L
            total = sum(preCount[i] for i in range(1, L)) % MOD
            # Add all stepping numbers of length L that are <= s
            total = (total + digitDP(s)) % MOD
            return total

        def minus_one(num_str: str) -> str:
            """
            Returns (num_str - 1) as a string. num_str is guaranteed to be >= "1".
            If the result is 0, returns "0".
            """
            if num_str == "0":
                return "0"  # shouldn't happen for valid input >= 1
            # Convert to list of digits
            arr = list(map(int, num_str))
            i = len(arr) - 1
            # Subtract one with borrow
            while i >= 0:
                if arr[i] > 0:
                    arr[i] -= 1
                    break
                else:
                    arr[i] = 9
                    i -= 1
            # If all digits become zero, return "0"
            # Otherwise, convert back to string ignoring leading zeros.
            while len(arr) > 1 and arr[0] == 0:
                arr.pop(0)
            res = "".join(map(str, arr))
            if res == "":
                return "0"
            return res

        # Main logic: countAllUpTo(high) - countAllUpTo(low - 1)
        low_minus_1 = minus_one(low)  # Handle the case low="1" -> "0"
        ans = countAllUpTo(high) - countAllUpTo(low_minus_1)
        ans %= MOD
        return ans