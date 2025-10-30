MOD = 10**9 + 7

# We'll precompute factorials and inverse factorials up to a limit.
# The maximum n needed in combinations is at most about 500.
_MAX = 500  
fact = [1] * (_MAX+1)
invfact = [1] * (_MAX+1)
for i in range(2, _MAX+1):
    fact[i] = fact[i-1] * i % MOD
invfact[_MAX] = pow(fact[_MAX], MOD-2, MOD)
for i in range(_MAX, 0, -1):
    invfact[i-1] = invfact[i] * i % MOD

def nCr(n, r):
    if r < 0 or r > n:
        return 0
    return fact[n] * invfact[r] % MOD * invfact[n-r] % MOD

class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        # Explanation:
        # We want to count arrays of length (zero+one) having exactly "zero" zeros
        # and "one" ones such that every contiguous subarray of length > limit 
        # contains both 0 and 1.
        # This condition is equivalent to saying that there is no block (run) 
        # of consecutive identical numbers longer than "limit".
        #
        # One way to count such arrays is to “cut” the sequence into alternating runs.
        # For a sequence that starts with a given digit, the sequence will be formed
        # by a series of runs that alternate: for example if it starts with 0:
        #   Option A: [0-run, 1-run, 0-run, 1-run, …, 0-run] (ends with 0)
        #   Option B: [0-run, 1-run, 0-run, 1-run, …, 1-run] (ends with 1)
        # In Option A, the number of runs of 0’s is a and the number of runs of 1’s is a-1.
        # In Option B, both counts are a.
        #
        # Similarly, if the sequence starts with 1, then the two cases are analogous.
        #
        # Now, if we want to distribute, say, X occurrences of a digit among r runs, 
        # with each run positive and at most "limit", then the number of ways is the number
        # of compositions of X into r parts each in [1, limit]. 
        #
        # We can count the number of compositions using an inclusion–exclusion formula:
        #
        # Let f(N, r) = number of ways to write N as a sum of r positive integers each at most limit.
        # For r>=1 (and define f(0,0)=1, f(N,0)=0 for N>0), one standard formula is:
        #
        #   f(N, r) = sum_{j=0}^{⌊(N-r)/limit⌋} (-1)^j * C(r, j) * C(N - j*limit - 1, r - 1)
        #
        # Of course, if N > r * limit then f(N, r)=0 automatically.
        #
        # Using this, the overall count is the sum of counts of arrays starting with 0 and with 1:
        #
        # Total = [count for start with 0] + [count for start with 1]
        # where:
        #   For start with 0:
        #     Option A (ends with 1): zeros appear in a runs and ones in a runs: ways = f(zero, a) * f(one, a)
        #     Option B (ends with 0): zeros in a runs and ones in (a-1) runs: ways = f(zero, a) * f(one, a-1)
        #   For start with 1: (symmetry)
        #     Option A (ends with 0): ones in a runs and zeros in a runs: ways = f(one, a) * f(zero, a)
        #     Option B (ends with 1): ones in a runs and zeros in (a-1) runs: ways = f(one, a) * f(zero, a-1)
        #
        # Thus, combining these we get:
        #   Total = 2 * sum_{a>=1} f(zero, a)*f(one, a)
        #           + sum_{a>=1} [ f(zero, a)*f(one, a-1) + f(one, a)*f(zero, a-1) ]
        #
        # We sum over possible number of runs a.
        #
        # We'll compute f(n, r) using the inclusion–exclusion formula,
        # caching the results to avoid recomputation.
        
        cache = {}
        def compositions(n, r):
            # returns the number of compositions of n into r positive integers each <= limit.
            # f(0, 0) = 1, and f(n, 0) = 0 for n > 0.
            if (n, r) in cache:
                return cache[(n, r)]
            if r == 0:
                cache[(n, r)] = 1 if n == 0 else 0
                return cache[(n, r)]
            # If too many parts or impossible sum condition:
            if r > n or n > r * limit:
                cache[(n, r)] = 0
                return 0
            S = n - r  # after subtracting 1 from each part.
            max_j = S // limit  # maximum j for which j*limit is not greater than S.
            total = 0
            for j in range(max_j + 1):
                # Using inclusion-exclusion
                # term = (-1)^j * C(r, j) * C(n - j*limit - 1, r - 1)
                sign = -1 if j % 2 == 1 else 1
                term = nCr(r, j) * nCr(n - j * limit - 1, r - 1) % MOD
                total = (total + sign * term) % MOD
            cache[(n, r)] = total % MOD
            return cache[(n, r)]
        
        # Now sum over possible numbers of runs. What is the maximum possible a?
        # For the digit with count X, we need a runs with 1 <= a <= X and X <= a*limit,
        # so a can range up to X. To cover both zeros and ones, we run a up to max(zero, one).
        ans = 0
        max_runs = max(zero, one)
        for a in range(1, max_runs + 1):
            term_same = compositions(zero, a) * compositions(one, a) % MOD  # for Option A in both patterns
            term_diff = compositions(zero, a) * compositions(one, a - 1) % MOD  # Option B for start with 0
            term_diff2 = compositions(one, a) * compositions(zero, a - 1) % MOD  # Option B for start with 1
            ans = (ans + 2 * term_same + term_diff + term_diff2) % MOD
        return ans % MOD