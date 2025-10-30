MOD = 10**9 + 7

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        # When k==0, only numbers that are already 1 (thus f(1)==0) are k-reducible.
        # Also, we count only positive integers less than n.
        # If n == 1 then answer is 0.
        n = int(s, 2)
        if n <= 1:
            return 0
        
        # Precompute f(m) -- the minimum number of operations needed to reduce m to 1
        # according to the rule: if m==1, f(m)=0; else f(m)=1+f(popcount(m))
        # We only need to compute for m up to max_possible, which is at most len(s)*1 (worst-case popcount)
        max_m = len(s) + 1  # a safe upper bound
        memo = {}
        def f(m):
            if m == 1:
                return 0
            if m in memo:
                return memo[m]
            # Compute popcount of m
            cnt = bin(m).count('1')
            val = 1 + f(cnt)
            memo[m] = val
            return val

        # We'll precompute f(m) for m from 1 to max_m.
        f_array = [0]*(max_m+1)
        for m in range(1, max_m+1):
            f_array[m] = f(m)
            
        # Our goal: count all positive integers x (< n) such that f(x) <= k.
        # Notice for any x >= 2, we have f(x) = 1 + f(popcount(x)).
        # So for x >= 2: f(x) <= k  <=> f(popcount(x)) <= k-1.
        # However, x==1 is a special case: f(1)==0.
        #
        # Thus, if k == 0, then the only x that qualify are those that are already 1.
        if k == 0:
            # count x==1 if 1 < n
            return 1 if 1 < n else 0

        # For x >= 2: condition is f(popcount(x)) <= k-1.
        # Let valid_set be the set of integers r (r>=1) such that f(r) <= k-1.
        valid_set = set()
        for r in range(1, max_m+1):
            if f_array[r] <= k-1:
                valid_set.add(r)
                
        # However, note that when x==1 we already have f(1)==0 (which is <= k)
        # and 1 is also covered by valid_set (since 1 is in valid_set because f(1)==0).
        # But in our counting below by "number of ones" (i.e. popcount(x)),
        # counting x with popcount = r sums over all x. So x==1 (which has exactly one set bit)
        # will be counted along with others.
        #
        # So our plan is to count for each r in valid_set the number of positive integers x (< n)
        # having exactly r ones in their binary representation.
        #
        # We need to do a digit DP (or combinatorics over positions) that counts the number of numbers
        # less than n (with binary representation same length as s, without leading zeros) that have exactly r ones.
        # Since s can be up to length 800, we'll implement a DP over the bits.
        
        L = len(s)
        # dp[pos][tight][ones] = number of ways to assign bits from pos to L-1,
        # where tight indicates that we are bound by the prefix of s (tight = 1 means the prefix so far is equal)
        # and we have chosen 'ones' count so far.
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(pos, tight, ones):
            if pos == L:
                # End of number, count only if at least one bit is set (i.e., number > 0)
                return 1 if ones >= 0 else 0  # we always count the constructed number (we don't want to count 0 anyway)
                # But our DP naturally does not form a number with all zeros because the original s has no leading zeros.
                # We'll handle the fact that we must require the number > 0 by our DP transitions.
            res = 0
            # Determine the upper limit for this bit.
            limit = int(s[pos]) if tight else 1
            for bit in range(limit+1):
                new_tight = tight and (bit == limit)
                # To avoid counting numbers with leading zero in a position where it matters,
                # note that since s has no leading zero, the MSB of any valid number is always 1.
                # We can allow the DP to generate numbers with leading zeros only if they are not used,
                # but then later we subtract the count for 0 (which is not positive).
                res = (res + dp(pos+1, new_tight, ones + bit)) % MOD
            return res
        
        # Use a combinatorial DP that counts numbers with exactly r ones.
        # We'll modify the DP to count ways and then subtract the count for 0 at the end.
        # Alternatively, we can do a DP that also tracks the condition whether we've already put a '1'
        # to guarantee nonzero. But it is simpler to count all numbers (including 0) and subtract the 0 case.
        #
        # Instead, we do a standard combinatorial approach:
        # Count for each r (from 0 to L) the number of integers with exactly r ones and <= n.
        # We can use digit DP that returns an array for each ones count.
        # Let's do that.
        
        # dp2[pos][tight][ones] returns count ways.
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp2(pos, tight, ones):
            if pos == L:
                return 1 if ones >= 0 else 0, ones  # not used: we need count per ones value.
            ways = [0]*(L+1)  # ways[i] = count ways with i ones, possible ones up to L.
            if pos == L:
                ways[ones] = 1
                return ways
            # Instead, we'll implement iterative DP instead.
            # However, because L can be as high as 800, it's simpler to go with an iterative Double DP.
            # We'll do that below.
            return None  # not used
        
        # We'll do iterative DP that collects counts for each ones count.
        # Let f[i][j] be the number of ways to assign bits from position i to L with exactly j ones,
        # with the constraint that the formed number is strictly less than the remaining suffix of s if required.
        # We implement a digit DP in a bottom-up manner.
        # We'll use a 2D DP: dp[pos][ones_count] for each state of "tight" separately.
        # Instead, we'll do a DP with dimension: dp[pos][tight][ones_count].
        
        # Initialize dp[0][1][0] = 1.
        dp_table = [[[0]*(L+1) for _ in range(2)] for _ in range(L+1)]
        dp_table[0][1][0] = 1
        
        for pos in range(L):
            cur_bit = int(s[pos])
            for tight in range(2):
                for ones in range(L+1):
                    ways = dp_table[pos][tight][ones]
                    if ways == 0:
                        continue
                    limit = cur_bit if tight==1 else 1
                    for bit in range(limit+1):
                        ntight = 1 if (tight==1 and bit==limit) else 0
                        dp_table[pos+1][ntight][ones+bit] = (dp_table[pos+1][ntight][ones+bit] + ways) % MOD

        # Now combine the counts from both tight states.
        count_by_ones = [0]*(L+1)
        for tight in range(2):
            for ones in range(L+1):
                count_by_ones[ones] = (count_by_ones[ones] + dp_table[L][tight][ones]) % MOD
        
        # dp counted the number 0 as well, which has 0 ones.
        count_by_ones[0] = (count_by_ones[0] - 1) % MOD  # remove the 0 number
        
        # Now, sum up counts for numbers with exactly r ones such that:
        # If the number is 1 (i.e. binary "1"), f(1)=0 <= k, and if n > 1, it is counted.
        # For numbers other than 1 (which have value >= 2), f(x) = 1+f(popcount(x)) <= k
        # <=> f(popcount(x)) <= k-1.
        # That means, if r = popcount(x), then we need f(r) <= k-1.
        ans = 0
        for r in range(1, L+1):
            if r in valid_set:
                ans = (ans + count_by_ones[r]) % MOD
        # Special case: the number 1 is counted correctly: its binary representation "1" has one set bit.
        # Our DP count_by_ones[1] includes it. However, note that sometimes our recurrence f(1)==0 and 0 <= k-1
        # is always true for k>=1. So it is already included.
        return ans % MOD
     
     
# You can include some test cases within a main block if needed.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.countKReducibleNumbers("111", 1))  # Expected output: 3
    # Example 2:
    print(sol.countKReducibleNumbers("1000", 2))  # Expected output: 6
    # Example 3:
    print(sol.countKReducibleNumbers("1", 3))     # Expected output: 0