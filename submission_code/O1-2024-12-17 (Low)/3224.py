class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        """
        We have n children in a line (positions 0 to n-1) and a sorted array sick of positions
        that are initially infected.  We want the number of possible ways ("infection sequences")
        in which all the remaining children (those not in sick) can become infected, subject to:
          - at most one new infection per time step,
          - a child can only become infected if it has an infected neighbor at that moment.
        We return the result modulo 10^9+7.

        -----------------------------------------------------------------------
        KEY IDEA / FORMULA:

        Let M = number of uninfected children = n - len(sick).

        Define "blocks" of consecutive uninfected children, separated by infected positions.
        Concretely, suppose sick = [s1, s2, ..., sk] is sorted.  We consider up to (k+1) blocks:
          Block 0: from position 0     to s1-1       (if s1 > 0)
          Block i: from s(i-1)+1       to s(i) - 1   (for i = 1..k-1, if that interval is non-empty)
          Block k: from s(k) + 1       to n-1        (if s(k) < n-1)

        Let L_i be the length (# children) of block i.

        Each block is "pinned" on one or both ends by an infected position (unless it touches
        the boundary of the entire line).  The crucial combinatorial fact is:

          1) If a block is pinned on BOTH ends (i.e. between two infected positions), it has
             2^(L-1) ways to infect internally, because at each infection step we can choose
             to infect from the leftmost or the rightmost uninfected child (once at least
             one end is infected).

          2) If a block is pinned on only ONE end or zero ends (boundary), then within that
             block there is exactly 1 possible way to infect all of its children in order
             (they must become infected outward from the pinned side).

        Meanwhile, all blocks can interleave in any fashion, because an infected child on one
        block does not constrain the infections in a different block (they are separated by
        at least one initially-infected child).  So if block i has L_i uninfected children that
        must appear in its own "internal" infection order (one of B_i possible ways), and block j
        has L_j children with B_j ways, then the total ways to combine (interleave) the infections
        of multiple blocks is the multinomial factor:

            (L1 + L2 + ... + Lb)! / (L1! * L2! * ... * Lb!)

        multiplied by (B_1 * B_2 * ... * B_b), where:
            B_i = 2^(L_i - 1) if block i is pinned on both ends and L_i > 0,
                  1 otherwise.

        Hence the final answer is:

          ways =  ( factorial(M) / (prod of factorial(L_i)) )  *  ( ∏ B_i )

        modulo 1e9+7.

        -----------------------------------------------------------------------
        EXAMPLE 1:
            n=5, sick=[0,4]
            Uninfected = [1,2,3], so M=3
            We have one block in the middle, pinned on both ends by infected 0 and 4.
            L=3 => B=2^(3-1)=4
            The interleaving factor with no other blocks is just 1
            => total=4

        EXAMPLE 2:
            n=4, sick=[1]
            Uninfected = [0,2,3], so M=3
            Blocks:
              Block0 = [0] pinned on one side (boundary+ infected 1) => L=1 => B=1
              Block1 = [2,3] pinned on one side (infected 1 + boundary) => L=2 => B=1
            Interleaving factor = factorial(3)/(1!*2!)=3
            product(B_i) = 1*1=1
            => total=3

        We implement this logic with appropriate factorial and inverse factorial computations.
        """

        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials up to n
        # so we can do combinations mod 10^9+7 efficiently.
        fact = [1]*(n+1)
        inv_fact = [1]*(n+1)
        
        for i in range(1, n+1):
            fact[i] = (fact[i-1]*i) % MOD
        
        # Fermat's little theorem for modular inverse: a^(MOD-2) mod MOD if MOD is prime
        inv_fact[n] = pow(fact[n], MOD-2, MOD)
        for i in reversed(range(n)):
            inv_fact[i] = (inv_fact[i+1]*(i+1)) % MOD
        
        def comb(a, b):
            if b<0 or b> a: return 0
            return (fact[a]*inv_fact[b]%MOD) * inv_fact[a-b] % MOD
        
        # Fast exponent of 2
        def fast2exp(e):
            return pow(2, e, MOD) if e>=0 else 1
        
        k = len(sick)
        # M = total uninfected
        M = n - k
        
        # Edge case: if no uninfected, answer is 1
        if M == 0:
            return 1
        
        # Build block lengths:
        blocks = []
        
        # Block 0 (left side) if sick[0] > 0
        if sick[0] > 0:
            blocks.append(sick[0])  # length from 0..(sick[0]-1)
        else:
            blocks.append(0)  # empty if first child is infected
        
        # Middle blocks
        for i in range(1, k):
            length_i = sick[i] - sick[i-1] - 1
            if length_i < 0:
                length_i = 0  # just a safeguard
            blocks.append(length_i)
        
        # Block k (right side) if sick[-1] < n-1
        if sick[-1] < n-1:
            blocks.append((n-1) - sick[-1])
        else:
            blocks.append(0)
        
        # Now we have (k+1) blocks (some may be zero-length).
        # B_i = 2^(L_i-1) if pinned on both ends and L_i>0, else 1.
        # pinned on both ends => the block is strictly between two infected children,
        # which happens for blocks i in [1..k-1].
        # block0 and block(k) are pinned on only one side if length>0.
        
        productB = 1
        b = len(blocks)  # should be k+1
        for i, L_i in enumerate(blocks):
            if L_i <= 0:
                continue
            # check if pinned on both ends:
            if 1 <= i <= k-1:  # interior block
                # it has two infected neighbors
                # so if L_i>0 => B=2^(L_i-1)
                ways_i = fast2exp(L_i-1)
            else:
                # pinned on one side
                ways_i = 1
            productB = (productB * ways_i) % MOD
        
        # ways to interleave = factorial(M) * product( inv_factorial(L_i) )
        waysInterleave = fact[M]
        for L_i in blocks:
            if L_i>0:
                waysInterleave = (waysInterleave * inv_fact[L_i]) % MOD
        
        ans = (waysInterleave * productB) % MOD
        return ans