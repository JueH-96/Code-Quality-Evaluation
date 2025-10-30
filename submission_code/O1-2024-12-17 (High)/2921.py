class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        # Digit-DP approach: count how many stepping numbers are in [1..X].
        # A "stepping number" is one where adjacent digits differ by 1,
        # and it does not have a leading zero (i.e., its decimal representation
        # does not start with 0).
        
        def count_stepping_up_to(X: str) -> int:
            """
            Returns the count of stepping numbers in the inclusive range [1..X].
            """
            # Convert string X to a list of digits
            digits = list(map(int, X))
            
            # Memo for DP: dp[(pos, last_digit, is_tight, started)] = count
            #   pos: current index in digits we are processing
            #   last_digit: the last chosen digit (0..9), meaningful only if started is True
            #   is_tight: indicates whether we are still bound by the prefix of X
            #   started: indicates whether we've chosen a nonzero digit yet (thus actually formed a number)
            dp = {}
            
            def dfs(pos: int, last_digit: int, is_tight: bool, started: bool) -> int:
                # If we've reached the end of the digit list:
                if pos == len(digits):
                    # Valid stepping number only if we've actually started
                    return 1 if started else 0
                
                key = (pos, last_digit, is_tight, started)
                if key in dp:
                    return dp[key]
                
                limit = digits[pos] if is_tight else 9
                result = 0
                
                for d in range(limit + 1):
                    nxt_tight = (is_tight and (d == limit))
                    
                    if not started:
                        # We haven't started forming a nonzero digit yet
                        if d == 0:
                            # Still not started (leading zeros)
                            result += dfs(pos + 1, last_digit, nxt_tight, False)
                        else:
                            # Start with a nonzero digit
                            result += dfs(pos + 1, d, nxt_tight, True)
                    else:
                        # Already started, need to enforce stepping: abs(d - last_digit) == 1
                        if abs(d - last_digit) == 1:
                            result += dfs(pos + 1, d, nxt_tight, True)
                    
                    result %= MOD
                
                dp[key] = result
                return result
            
            return dfs(0, 0, True, False)
        
        # Counts stepping numbers in [L..R] = count_stepping_up_to(R) - count_stepping_up_to(L-1)
        # Handle the case when L = 1 separately (then L-1 = 0).
        
        # Convert low to a big integer so we can do low-1
        low_val = int(low)
        
        high_count = count_stepping_up_to(high)  # stepping numbers up to high
        if low_val > 1:
            # stepping numbers up to (low-1)
            low_minus_one = str(low_val - 1)
            low_count = count_stepping_up_to(low_minus_one)
            answer = (high_count - low_count) % MOD
        else:
            # low == 1
            answer = high_count % MOD
        
        return answer