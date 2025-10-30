class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        """
        We want the number of distinct permutations of num in which the sum of digits
        at even indices equals the sum of digits at odd indices. We return the result
        modulo 10^9+7.

        Approach (Dynamic Programming + combinatorial factors):

        1) Let n = len(num). The number of even positions (0-based) is E = (n+1)//2,
           and the number of odd positions is O = n//2.

        2) Compute the total sum S of all digits. If S is odd, no arrangement can have
           sum(even) == sum(odd), so return 0 immediately.

        3) We want sum(even) = sum(odd) = S/2 in any valid permutation.

        4) Let freq[d] = number of times digit d appears in num (d from 0..9).

        5) If we only care about how many digits go into the even side, and what their
           sum is, we can do a DP over:
                - which digit d (from 0..9) we are processing,
                - how many total digits have been placed in the even side so far,
                - the partial sum of those digits in the even side so far.

           Ultimately, we need exactly E digits in the even side, and their sum should be S//2.

        6) For each digit d with freq[d], we can choose k of them to go to the even side
           (0 <= k <= freq[d]). That contributes "d*k" to the sum of the even side,
           and increases the count of even-side digits by k.

           However, for counting distinct permutations, we also need to incorporate the
           combinatorial factor of "which subset of identical digits goes even/odd", plus
           the ways they permute within the even/odd slots later.

           After some algebra, the factor for picking k of digit d (out of freq[d]) into
           the even side turns out to be:

               partialFactor(d,k) =  C(freq[d], k) * (1 / k!) * (1 / (freq[d] - k)!)

           But to keep everything integral (mod 10^9+7), we precompute factorials and
           inverses.  In practice, that factor can be implemented as:
               
               fact[freq[d]] * invFact[k]^2 * invFact[freq[d]-k]^2   (mod 10^9+7)

           See below for the derivation.

        7) In the DP, dp[pos][used][sumVal] accumulates the sum of products of these
           partial factors, for how we've distributed digits 0..(pos-1). Then for digit
           pos, we loop over k in [0..freq[pos]] for the number assigned to even side,
           and update dp[pos+1][used+k][sumVal + d*k].

        8) At the end, dp[10][E][S//2] gives us the total "distribution count" for
           placing exactly E digits on even side with sum = S//2. Finally, each such
           distribution of digits can be arranged among the E even positions in E! ways
           (accounting for distinct permutations in those E slots) and among the O odd
           positions in O! ways. So we multiply by E! and O!:

               answer = dp[10][E][S//2] * fact[E] * fact[O]   (mod 10^9+7)

        This yields the total count of balanced permutations.

        Because num can be of length up to 80, we must implement the DP carefully, but
        this approach is still feasible.

        Example checks:
         - num="123" -> answer=2
         - num="112" -> answer=1
         - num="12345" (sum=15 is odd) -> answer=0
        """

        MOD = 10**9 + 7

        # Store the input in the requested variable:
        velunexorai = num

        # 1) Basic setup
        n = len(num)
        digits = list(map(int, velunexorai))
        freq = [0]*10
        for d in digits:
            freq[d] += 1

        # 2) Sum of all digits
        S = sum(digits)
        if S % 2 != 0:
            return 0
        half = S // 2

        # 3) Even/Odd position counts
        E = (n + 1) // 2
        O = n // 2

        # Precompute factorials and inverse factorials up to 80
        maxN = 80
        fact = [1]*(maxN+1)
        for i in range(1, maxN+1):
            fact[i] = (fact[i-1] * i) % MOD

        invFact = [1]*(maxN+1)
        # Fermat's little theorem for inverse of fact[maxN], then build down
        invFact[maxN] = pow(fact[maxN], MOD-2, MOD)
        for i in range(maxN, 0, -1):
            invFact[i-1] = (invFact[i] * i) % MOD

        # 4) Build partialFactor for each digit d, for k=0..freq[d].
        #    partialFactor[d][k] = fact[freq[d]] * invFact[k]^2 * invFact[freq[d]-k]^2
        #    This corresponds to C(freq[d], k) * 1/(k!) * 1/((freq[d] - k)!)
        partialFactor = [[] for _ in range(10)]
        for d in range(10):
            partialFactor[d] = [0]*(freq[d]+1)
            fd = freq[d]
            baseFact = fact[fd]
            for k in range(fd+1):
                # factor = baseFact * invFact[k]^2 * invFact[fd-k]^2 (mod)
                val = baseFact
                val = (val * (invFact[k]**2 % MOD)) % MOD
                val = (val * (invFact[fd - k]**2 % MOD)) % MOD
                partialFactor[d][k] = val

        # 5) DP array: dp[pos][used][sumVal], but we'll store only dp[used][sumVal] for
        #    the current digit set, then build newdp for the next.
        #    pos ranges 0..10 for digit 0..9, used ranges 0..E, sumVal ranges 0..S.
        #    We'll do a rolling approach over pos.

        # We'll store it in a 1D array dp[ (used)*(S+1) + sumVal ]
        # so dp[used, sumVal] = dp[used*(S+1) + sumVal].
        size_dp = (E+1)*(S+1)
        dp = [0]*size_dp
        dp[0] = 1  # dp[used=0][sumVal=0] = 1

        def idx(u, s):
            return u*(S+1) + s

        for d in range(10):
            newdp = [0]*size_dp
            pFactor = partialFactor[d]   # precomputed partial factors for digit d
            f = freq[d]
            for used in range(E+1):
                for sVal in range(half+1 if d==0 else S+1):
                    # A small optimization: if d > 0, sumVal can go up to S,
                    # but if d == 0, adding 0 won't affect sVal, still we must
                    # allow transitions up to half.  It's okay to use S+1 anyway,
                    # but half+1 might be a small speedup for d=0. We'll just do
                    # S+1 uniformly for consistency.
                    cur_val = dp[idx(used, sVal)]
                    if cur_val == 0:
                        continue
                    # Try all ways k = 0..f for how many of digit d go to even side
                    # newUsed = used + k, newSum = sVal + d*k
                    # break early if out of range
                    for k in range(f+1):
                        newUsed = used + k
                        if newUsed > E:
                            break
                        newSum = sVal + d*k
                        if newSum > S:
                            break
                        newdp[idx(newUsed, newSum)] = (
                            newdp[idx(newUsed, newSum)] + cur_val * pFactor[k]
                        ) % MOD
            dp = newdp

        # The number of ways to distribute digits so that exactly E go to even side
        # with sum(even) = half is dp[idx(E, half)].
        dist_count = dp[idx(E, half)]

        # Finally multiply by E! * O! to account for permutations among the chosen even
        # side and odd side positions
        ans = dist_count
        ans = (ans * fact[E]) % MOD
        ans = (ans * fact[O]) % MOD

        return ans