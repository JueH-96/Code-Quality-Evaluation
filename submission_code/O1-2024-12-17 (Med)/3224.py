class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        """
        We have n children in a line, with some subset (sick) infected at the start.
        A child becomes infected only after at least one of its neighbors is infected,
        and exactly one new child can be infected at each time step. We want the total
        number of possible sequences in which all the previously-uninfected children
        can become infected. Return that count modulo 10^9+7.

        --------------------------------------------------------------------
        Key Idea (Combinatorial / "Block + Interleaving" argument):
        
        1) Mark which children are initially infected. Then scan left-to-right to find
           each maximal contiguous block of uninfected children. Such a block is
           "bounded" on each side by either (a) an infected child or (b) the boundary
           of the line (index -1 or n).
           
           Example: If n=5 and sick=[0,4], the uninfected positions are 1,2,3 which
           form a single block [1..3], bounded by infected on both ends.
           
           As another example, n=4 and sick=[1] => uninfected are 0,2,3. The line
           looks like [0] - [1 infected] - [2,3]. This gives two blocks:
               • [0..0] (to the left of index 1),
               • [2..3] (to the right of index 1).

        2) Within a single block of length L:
           - If it is bounded on both ends by infected children, then that block
             can be "infected from both ends," giving 2^(L-1) possible ways to
             order infections within that block. 
             (Reason: at each step, you can infect either the leftmost uninfected
              or the rightmost uninfected in that block, until it is fully infected.)
           
           - Otherwise (bounded on only one side by an infected child, or by an infected
             child on one side and the boundary on the other), there is exactly 1 way
             to infect the block: it proceeds in a forced chain from the infected side
             inward. So that contributes 1 to the count of internal orders for that block.

        3) Once we know the count of "local orders" e_i for each block i, we must also
           count all ways to interleave the infections among these blocks. If there
           are k blocks of lengths L1, L2, ..., Lk (summing to M = n - len(sick)
           total uninfected children), then the blocks form disjoint "partial orders."
           
           The number of linear extensions of a union of k disjoint partial orders
           of sizes L1..Lk (with e_i ways each) is:
               
               (e_1 * e_2 * ... * e_k) * (M! / (L1! * L2! * ... * Lk!))
               
           because we can choose any interleaving of those L1 + L2 + ... + Lk = M
           infection events, and each block internally has e_i ways.

        4) Hence the final formula is:
             
               Answer = 
                 ( ∏ e_i ) 
                 × 
                 (
                   M! × inv( L1! ) × inv( L2! ) × ... × inv( Lk! )
                 )
             
           all computed modulo 10^9+7.

        We implement this by:
          - Building an array "infected[]" of length n marking which are initially infected.
          - Scanning for contiguous intervals [s..e] of uninfected children.
          - For each interval of length L = e-s+1, decide if it's bounded by infected
            on both sides or not. Bounded on both => e_i = 2^(L - 1); otherwise e_i = 1.
          - Collect the lengths L_i in a list. Sum them to M.
          - Multiply all e_i to get productE.
          - Then compute final = productE * ( factorial(M) / ( factorial(L1) * ... * factorial(Lk) ) ) mod.

        This runs in O(n) plus O(n) for precomputing factorials (which is feasible for n up to 10^5).
        """

        MOD = 10**9 + 7
        
        #----------------------------------------------------------------------------
        # 1) Build a quick lookup: infected[i] = True/False
        #----------------------------------------------------------------------------
        infected_arr = [False]*n
        for pos in sick:
            infected_arr[pos] = True
        
        #----------------------------------------------------------------------------
        # 2) Precompute factorials and inverses to handle combinations quickly
        #    Also precompute powers of 2 up to n.
        #----------------------------------------------------------------------------
        maxN = n
        fac = [1]*(maxN+1)        # fac[i] = i! mod
        for i in range(1, maxN+1):
            fac[i] = (fac[i-1] * i) % MOD
        
        # Fermat inverse of x under MOD (since MOD is prime) is pow(x, MOD-2, MOD).
        inv_fac = [1]*(maxN+1)    # inv_fac[i] = (i!)^-1 mod
        inv_fac[maxN] = pow(fac[maxN], MOD-2, MOD)
        for i in reversed(range(maxN)):
            inv_fac[i] = (inv_fac[i+1] * (i+1)) % MOD
        
        # Function to compute nCk mod
        def nCk(n, k):
            if k<0 or k>n: 
                return 0
            return (fac[n] * inv_fac[k] % MOD) * inv_fac[n-k] % MOD
        
        # Precompute powers of 2
        pow2 = [1]*(maxN+1)
        for i in range(1, maxN+1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        #----------------------------------------------------------------------------
        # 3) Identify contiguous blocks of uninfected children.
        #    We'll store (start, end, length).
        #----------------------------------------------------------------------------
        intervals = []
        i = 0
        while i < n:
            if not infected_arr[i]:
                start = i
                while i < n and not infected_arr[i]:
                    i += 1
                end = i - 1  # i has advanced just past the block
                intervals.append((start, end, end - start + 1))
            else:
                i += 1
        
        #----------------------------------------------------------------------------
        # 4) For each block, compute the "internal ways" e_i, then keep track of L_i.
        #----------------------------------------------------------------------------
        product_e = 1   # product of local ways e_i
        lengths = []
        
        for (start, end, length) in intervals:
            # If an interval is of length L (start..end),
            # check if it's bounded by infected on left and right
            leftBoundInfected  = (start > 0   and infected_arr[start-1])
            rightBoundInfected = (end < n-1   and infected_arr[end+1])
            
            if length == 0:
                # no uninfected child, skip
                continue
            
            lengths.append(length)
            
            if leftBoundInfected and rightBoundInfected:
                # both ends infected => 2^(L-1) ways
                ways_block = pow2[length-1] if length > 0 else 1
            else:
                # single end (or no end infected, which means the infection
                # must come from the other side) => exactly 1 way
                ways_block = 1
            
            product_e = (product_e * ways_block) % MOD
        
        # M = total number of uninfected children
        M = sum(lengths)
        if M == 0:
            # Means all children were already infected, though the problem states
            # at least one was uninfected if sick.length < n. But let's be safe:
            return 1  # no new infections => exactly 1 "empty" sequence
        
        #----------------------------------------------------------------------------
        # 5) Combine (interleave) the blocks: multiply by M! / (L1! * L2! * ... * Lk!)
        #----------------------------------------------------------------------------
        # M! mod
        all_fact = fac[M]
        # divide by each L_i!
        denom = 1
        for L in lengths:
            denom = (denom * fac[L]) % MOD
        
        # final answer = product_e * ( M! * inv(denom) ) mod
        denom_inv = pow(denom, MOD-2, MOD)  # or use inv_fac logic carefully
        answer = product_e
        answer = (answer * all_fact) % MOD
        answer = (answer * denom_inv) % MOD
        
        return answer