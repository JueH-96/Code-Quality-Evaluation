MOD = 10**9 + 7

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        # count numbers in [low, high] that are stepping numbers.
        # We do this by computing count(≤ high) - count(≤ low-1) mod MOD.
        #
        # We'll use a digit-DP routine that counts valid numbers 
        # (without any leading digits) that are stepping numbers.
        #
        # A stepping number is defined so that once its first non‐zero digit 
        # has been chosen, each subsequent digit (in the number’s canonical representation)
        # must be equal to either (prev+1) or (prev-1). We allow numbers that have fewer digits 
        # than the given limit (they are represented with leading zeros in DP, but we only count 
        # those that eventually “start” with a nonzero digit).
        
        def count(x: str) -> int:
            digits = list(map(int, x))
            n = len(digits)
            
            # dp(pos, last, started, tight) returns count of valid numbers from position pos onward.
            # last: the last digit chosen, if started==True; if not started then last is irrelevant.
            # started: whether we have picked any nonzero digit (and thus begun our number).
            # tight: whether we are still constrained by the prefix of x.
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dfs(pos: int, last: int, started: bool, tight: bool) -> int:
                if pos == n:
                    # At end of the number, count only if a number has started.
                    return 1 if started else 0
                
                total = 0
                up = digits[pos] if tight else 9
                # We iterate over all possible digits 0..up.
                for d in range(0, up + 1):
                    ntight = tight and (d == up)
                    
                    if not started:
                        if d == 0:
                            # Still haven't started. (We can take leading zero here.)
                            # Note: If we never start, then at the end the number will be 0,
                            # but 0 is not a valid stepping number (and low>=1 by problem constraints),
                            # so our base case returns 0 when not started.
                            total = (total + dfs(pos + 1, -1, False, ntight)) % MOD
                        else:
                            # Now the number "starts" with a non-zero digit.
                            total = (total + dfs(pos + 1, d, True, ntight)) % MOD
                    else:
                        # Already started, so we must obey the stepping property.
                        # The next digit is forced: it must be equal to last-1 or last+1 (if valid).
                        candidates = []
                        if last - 1 >= 0:
                            candidates.append(last - 1)
                        if last + 1 <= 9:
                            candidates.append(last + 1)
                        # Only allow d that is in the allowed candidate set.
                        if d in candidates:
                            total = (total + dfs(pos + 1, d, True, ntight)) % MOD
                return total % MOD
            
            return dfs(0, -1, False, True) % MOD
        
        # helper to subtract 1 from a numeric string (x >= "1")
        def dec_str(s: str) -> str:
            # convert to list of digits for easier manipulation.
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0:
                if s_list[i] != '0':
                    s_list[i] = str(int(s_list[i]) - 1)
                    break
                else:
                    s_list[i] = '9'
                    i -= 1
            # remove any leading zeros that might occur (but if the answer becomes "0", return "0")
            res = ''.join(s_list).lstrip('0')
            return res if res != "" else "0"
        
        # The answer: count numbers <= high minus count numbers <= (low - 1)
        low_minus_one = dec_str(low)
        res = (count(high) - count(low_minus_one)) % MOD
        return res