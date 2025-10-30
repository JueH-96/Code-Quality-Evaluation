class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # We'll try both target final values: 0 and 1
        # The operations “invert prefix” and “invert suffix” are commutative, and
        # can be applied in any order. In our formulation we “choose” a set P of indices where
        # we perform the prefix‐flip (cost = i+1 for index i in P) and a set Q of indices where
        # we perform the suffix‐flip (cost = n-i for index i in Q).
        # A prefix flip at index i affects indices [0, i] and a suffix flip at index j affects indices [j, n-1].
        # Thus the total parity affecting index k is:
        #   parity(k) = (number of prefix flips with i >= k) + (number of suffix flips with j <= k) (mod 2).
        # We need parity(k) = (target XOR int(s[k])). Let:
        #   a[k] = target XOR int(s[k]).
        #
        # Instead of choosing flips arbitrarily, we can note that a suffix flip operation applied at index j (j>=1)
        # can be “linked” with the prefix flip decision at index j-1. In fact, by a telescoping argument,
        # it is possible to show that there exists an optimal solution that is determined completely by
        # a binary vector P[0..n-1] where:
        #   • For each index j (0 <= j <= n-2), we “force” Q[j+1] = P[j] XOR (a[j] XOR a[j+1]).
        #   • Q[0] is forced by the overall parity condition: Q[0] = a[0] XOR (sum_{j=0}^{n-1}P[j] mod2).
        #
        # Then the cost becomes:
        #   cost = Sum_{j=0}^{n-1} [ P[j]*(j+1) ]  +  Sum_{j=0}^{n-2}[ (P[j] XOR d[j])*(n - j - 1) ]
        #          + (a[0] XOR (total_parity))*(n)
        # where d[j] = a[j] XOR a[j+1].
        #
        # We now choose the binary vector P (of length n) to minimize the total cost.
        # We'll solve this with a DP that runs over positions 0..n-1 with state equal to the parity so far, i.e.
        # p = (P[0] + P[1] + ... + P[j-1]) mod 2.
        # Then when deciding P[j]=b (0 or 1), we add the cost for that choice:
        #   cost_prefix = (j+1)*b.
        #   Additionally, if j <= n-2, we add cost_suffix = (n - j - 1) if (b XOR d[j])==1, else 0.
        #
        # After processing all indices, let the overall parity be p_total.
        # Then we must add the cost for Q[0], which is: if a[0] XOR p_total == 1, then add cost n.
        #
        # We do this for target = 0 and target = 1, and take the minimum.
        import math

        def compute_for_target(target: int) -> int:
            # Build array a: a[i] = target XOR int(s[i])
            a = [target ^ (ord(ch) - ord('0')) for ch in s]
            # Build differences: d[i] = a[i] XOR a[i+1] for i=0..n-2.
            d = [a[i] ^ a[i+1] for i in range(n-1)]
            
            # dp[j][p] = minimum cost after processing indices 0..j-1, and current parity = p (0 or 1)
            dp = [[math.inf]*2 for _ in range(n+1)]
            dp[0][0] = 0
            
            for j in range(n):
                for p in (0, 1):
                    if dp[j][p] == math.inf:
                        continue
                    # decide P[j] = b in {0,1}
                    for b in (0, 1):
                        np = p ^ b
                        cost_here = (j+1) * b  # cost for prefix flip at index j if chosen
                        # For j from 0 to n-2, there is an associated cost for Q[j+1] = P[j] XOR d[j]
                        if j < n-1:
                            # cost of the suffix flip operation at index j+1 if needed
                            if (b ^ d[j]) == 1:
                                cost_here += (n - j - 1)
                        # update dp for next index
                        if dp[j][p] + cost_here < dp[j+1][np]:
                            dp[j+1][np] = dp[j][p] + cost_here
            
            best = math.inf
            # After processing all indices, overall parity is p_total.
            # We add the cost for Q[0] which is: if (a[0] XOR p_total)==1, then cost += n.
            for p_total in (0, 1):
                extra = n if (a[0] ^ p_total) == 1 else 0
                best = min(best, dp[n][p_total] + extra)
            return best

        # Evaluate both targets and return minimum cost.
        return min(compute_for_target(0), compute_for_target(1))