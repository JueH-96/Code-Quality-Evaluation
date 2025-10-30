class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # We'll use digit DP to count, via a helper function that returns the count
        # of beautiful integers (those that have an equal number of even and odd digits 
        # and are divisible by k) in [0, x]. Then our answer is f(high) - f(low-1).
        #
        # Key points:
        # 1. A number is beautiful if the count of even digits equals the count of odd digits.
        #    Notice that for any nonzero number, the actual representation is used; 
        #    leading zeros do not count. (Exception: 0 is represented as "0" but that won't be considered 
        #    since low>=1.)
        # 2. Only numbers with an even number of digits (in their usual representation) can satisfy 
        #    the condition. However, in our DP we allow leading zeros and ensure that when we have started 
        #    the number (non-leading part) we update our counts.
        #
        # DP state parameters:
        #   pos    : current digit index (0-indexed) in the string representation of x.
        #   tight  : boolean flag indicating if we are restricted by the prefix (i.e. so far we've chosen 
        #            digits that match the prefix of x).
        #   started: boolean flag indicating whether we have already chosen a nonzero digit (i.e. the start
        #            of the number has been reached).
        #   diff   : current difference (number_of_even - number_of_odd) among the digits counted so far.
        #   rem    : current remainder mod k of the number built so far.
        #
        # Transitions:
        # - If we haven't started (leading zeros), we can either choose 0 and remain not started,
        #   or choose a nonzero digit and then mark started as True.
        # - If started is True, every chosen digit is appended into the number.
        # - For each digit d chosen:
        #       If started==False and d==0 -> we do not update diff or rem.
        #       If started==False and d > 0, or if started is True:
        #           update started=True, update diff: diff + 1 if d is even, diff - 1 if d is odd,
        #           and update rem accordingly.
        #
        # Final condition:
        #  At the end (pos == len(s)), if we have started (so that the number is not just leading zeros)
        #  and the diff is exactly 0, and the remainder rem is 0 mod k, then we count that number.
        #  (Note: if we never started then the number is 0, whose representation "0" has 1 digit, 
        #   not satisfying equal counts.)
            
        from functools import lru_cache
        
        def count_beautiful(x: int) -> int:
            # Given x, count how many beautiful integers in [0, x].
            s = str(x)
            n = len(s)
            
            @lru_cache(maxsize=None)
            def dp(pos: int, tight: bool, started: bool, diff: int, rem: int) -> int:
                if pos == n:
                    # Only count if we have started and the digit counts match (diff==0)
                    # (Since if we haven't started, the number is 0, but "0" is a one-digit number,
                    #  so it doesn't satisfy parity condition.)
                    if started and diff == 0 and rem == 0:
                        return 1
                    return 0
                    
                ans = 0
                limit = int(s[pos]) if tight else 9
                
                for dig in range(0, limit+1):
                    new_tight = tight and (dig == limit)
                    
                    if not started:
                        if dig == 0:
                            # still not started: do not update diff or remainder
                            ans += dp(pos+1, new_tight, False, diff, rem)
                        else:
                            # now we are starting the number with a nonzero digit.
                            new_started = True
                            # Update diff: if the digit is even, add +1, if odd, subtract 1.
                            new_diff = diff + (1 if (dig % 2 == 0) else -1)
                            new_rem = (rem * 10 + dig) % k
                            ans += dp(pos+1, new_tight, new_started, new_diff, new_rem)
                    else:
                        # Already started, so each digit counts.
                        new_diff = diff + (1 if (dig % 2 == 0) else -1)
                        new_rem = (rem * 10 + dig) % k
                        ans += dp(pos+1, new_tight, True, new_diff, new_rem)
                return ans
            
            return dp(0, True, False, 0, 0)
        
        # Use our helper to count the numbers in [low, high]:
        return count_beautiful(high) - count_beautiful(low - 1)