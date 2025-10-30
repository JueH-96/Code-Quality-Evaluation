class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        # Helper function to subtract one from a string representing a positive integer.
        # Assumes n >= "1".
        def subtract_one(n: str) -> str:
            # Convert to list of digits
            arr = list(n)
            i = len(arr) - 1
            
            # Subtract one from the last digit, handling borrow if needed
            while i >= 0:
                if arr[i] == '0':
                    arr[i] = '9'
                    i -= 1
                else:
                    arr[i] = str(int(arr[i]) - 1)
                    break
            
            # If the leading digit becomes '0', remove it (e.g. "1000" -> "999").
            # If the entire string is "0" then we just return "0".
            if arr[0] == '0' and len(arr) > 1:
                # remove leading zeros
                j = 0
                while j < len(arr) and arr[j] == '0':
                    j += 1
                arr = arr[j:] if j < len(arr) else ['0']
            
            return "".join(arr)

        # Digit DP approach to count how many stepping numbers are <= n.
        def count_steps_upto(n_str: str) -> int:
            # Special case: if n_str == "0", then return 0
            # (there are no valid stepping numbers <= 0 with no leading zeros)
            if n_str == "0":
                return 0

            digits = list(map(int, n_str))
            length = len(digits)
            from functools import lru_cache

            @lru_cache(None)
            def dfs(pos, prev, is_tight, leading):
                # pos: current index in digits
                # prev: the previous digit chosen (or -1 if none chosen yet)
                # is_tight: bool indicating if we have to stay within digits[pos] limit
                # leading: bool indicating if we have not chosen any nonzero digit yet

                if pos == length:
                    # If we've placed at least one digit (leading=False), it's valid
                    return 0 if leading else 1

                limit = digits[pos] if is_tight else 9
                total = 0
                for dig in range(limit + 1):
                    if leading:
                        # We haven't chosen a nonzero digit yet
                        if dig == 0:
                            # Still leading zeros, so continue
                            total += dfs(pos + 1, -1, is_tight and dig == limit, True)
                        else:
                            # First nonzero digit
                            total += dfs(pos + 1, dig, is_tight and dig == limit, False)
                    else:
                        # We already have a previous digit, enforce stepping rule
                        if abs(dig - prev) == 1:
                            total += dfs(pos + 1, dig, is_tight and dig == limit, False)
                    total %= MOD

                return total

            return dfs(0, -1, True, True)

        # Main logic: count stepping numbers in [low, high] = count <= high - count <= (low-1)
        # Handle when low == "1", then low-1 = "0"
        low_minus_1 = subtract_one(low) if low != "1" else "0"
        ans = count_steps_upto(high) - count_steps_upto(low_minus_1)
        ans %= MOD
        return ans