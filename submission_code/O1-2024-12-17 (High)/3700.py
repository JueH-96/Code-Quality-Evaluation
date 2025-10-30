class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        mod = 10**9 + 7
        
        n = len(nums)
        # Edge case: if n < 5, answer is 0
        if n < 5:
            return 0
        
        #-----------------------------------------------------------------------
        # 1) Precompute nCr (binomial coefficients) up to n=1000 for fast C(a,b).
        #    We will need these for freq(x) = 3, 4, 5 scenarios.
        #-----------------------------------------------------------------------
        maxN = 1000
        fact = [1]*(maxN+1)
        invfact = [1]*(maxN+1)
        for i in range(1, maxN+1):
            fact[i] = fact[i-1]*i % mod
        
        # Fermat's little theorem for inverse of fact[maxN] mod p
        invfact[maxN] = pow(fact[maxN], mod-2, mod)
        for i in reversed(range(maxN)):
            invfact[i] = (invfact[i+1]*(i+1)) % mod
        
        def nCr(a, b):
            if b<0 or b> a: 
                return 0
            return (fact[a]*invfact[b]%mod)*invfact[a-b]%mod
        
        #-----------------------------------------------------------------------
        # 2) Precompute, for each distinct value v, the positions it occupies.
        #    This lets us quickly find how many times v appears to the left/right
        #    of a given index j by binary-searching in positions[v].
        #-----------------------------------------------------------------------
        from collections import defaultdict
        positions = defaultdict(list)
        for i, val in enumerate(nums):
            positions[val].append(i)
        
        def count_left(v, j):
            # How many positions of value v are strictly less than j
            # We can do a binary search in positions[v].
            arr = positions[v]
            # bisect_left to find how many indices < j
            import bisect
            return bisect.bisect_left(arr, j)
        
        def count_right(v, j):
            # How many positions of value v are strictly greater than j
            # If total freq is len(arr), and idx = bisect_right(arr, j),
            # then count is len(arr) - idx
            arr = positions[v]
            import bisect
            idx = bisect.bisect_right(arr, j)
            return len(arr) - idx
        
        #-----------------------------------------------------------------------
        # 3) Helper to count freq(x)=2 subsequences for a given j (middle = j).
        #
        #    We must pick exactly 4 indices among left side [0..j-1] and right
        #    side [j+1..n-1], with exactly one more x among them (so total x=2),
        #    and ensure no other value appears twice (so no tie).
        #
        #    Concretely:
        #      - Either (left-pair has (x, v) and right-pair has two distinct
        #        values neither of which is x or v),
        #        or
        #      - (right-pair has (x, v) and left-pair has two distinct values
        #        neither of which is x or v).
        #
        #    We'll build pair-counts in O(L^2) and O(R^2) where L=j, R=n-1-j.
        #-----------------------------------------------------------------------
        from collections import defaultdict
        
        def count_freq2_for_j(j):
            x = nums[j]
            L = j
            R = n - 1 - j
            if L < 2 and R < 2:
                return 0
            # Build structures for left side
            # cat1_left[v] = number of pairs (i1, i2) in [0..j-1] with exactly (x,v)
            # pairsLeftNoX[(v1,v2)] = number of pairs with v1<v2, both != x, distinct
            cat1_left = defaultdict(int)
            pairsLeftNoX = defaultdict(int)
            
            for i1 in range(L):
                for i2 in range(i1+1, L):
                    v1, v2 = nums[i1], nums[i2]
                    if v1 == v2:
                        # skip if same value => that would cause freq(v1)=2 if v1!=x, or (x,x) if v1=x
                        # not allowed in freq(x)=2 scenario
                        continue
                    # sort (v1,v2) to store consistently
                    if v1 > v2:
                        v1, v2 = v2, v1
                    # now v1 <= v2
                    if v1 == x and v2 != x:
                        cat1_left[v2] += 1
                    elif v1 != x and v2 == x:
                        # after sorting, this case won't happen if x < v1 originally,
                        # but we'll handle it anyway
                        cat1_left[v1] += 1
                    elif v1 != x and v2 != x:
                        # both distinct from x, also v1 != v2
                        pairsLeftNoX[(v1,v2)] += 1
            
            # Precompute sums for the left side
            sumLeftNoX = 0
            freqLeftNoX = defaultdict(int)
            for (v1,v2), c in pairsLeftNoX.items():
                sumLeftNoX = (sumLeftNoX + c) % mod
                freqLeftNoX[v1] += c
                freqLeftNoX[v2] += c
            
            # Build structures for right side
            cat1_right = defaultdict(int)
            pairsRightNoX = defaultdict(int)
            
            for i1 in range(j+1, n):
                for i2 in range(i1+1, n):
                    v1, v2 = nums[i1], nums[i2]
                    if v1 == v2:
                        # skip if same value => that would cause freq(v1)=2 if v1!=x
                        continue
                    if v1 > v2:
                        v1, v2 = v2, v1
                    if v1 == x and v2 != x:
                        cat1_right[v2] += 1
                    elif v1 != x and v2 == x:
                        cat1_right[v1] += 1
                    elif v1 != x and v2 != x:
                        pairsRightNoX[(v1,v2)] += 1
            
            sumRightNoX = 0
            freqRightNoX = defaultdict(int)
            for (v1,v2), c in pairsRightNoX.items():
                sumRightNoX = (sumRightNoX + c) % mod
                freqRightNoX[v1] += c
                freqRightNoX[v2] += c
            
            # Distribution #1: left side picks (x,v), right side picks two-distinct not-x not-v
            dist1 = 0
            for v, c in cat1_left.items():
                # from the right side, we want pairs that do not contain x or v
                # we already excluded x within pairsRightNoX
                # we only need to exclude pairs that contain v
                # total pairs in rightNoX is sumRightNoX, pairs that contain v is freqRightNoX[v]
                cnt = (sumRightNoX - freqRightNoX[v]) % mod
                dist1 = (dist1 + c*cnt) % mod
            
            # Distribution #2: right side picks (x,v), left side picks two-distinct not-x not-v
            dist2 = 0
            for v, c in cat1_right.items():
                # from the left side, we want two-distinct not-x not-v
                cnt = (sumLeftNoX - freqLeftNoX[v]) % mod
                dist2 = (dist2 + c*cnt) % mod
            
            return (dist1 + dist2) % mod
        
        #-----------------------------------------------------------------------
        # 4) Putting it all together. For each j, we compute:
        #    - ways_2(j): freq(x)=2 with unique mode
        #    - ways_3(j): freq(x)=3
        #    - ways_4(j): freq(x)=4
        #    - ways_5(j): freq(x)=5
        #
        #    Then sum over j (where j is the middle index).
        #-----------------------------------------------------------------------
        
        ans = 0
        
        for j in range(n):
            # We need at least 2 indices on the left and 2 on the right
            if j < 2 or j > n-3:
                # No room to pick 2 on each side
                continue
            
            x = nums[j]
            leftCountX = count_left(x, j)      # how many x in [0..j-1]
            rightCountX = count_right(x, j)    # how many x in [j+1..n-1]
            L = j
            R = (n-1-j)
            
            # freq(x)=5 => subsequence of all x's
            # means we pick 2 from left side (all x) and 2 from right side (all x)
            # number of ways = C(leftCountX, 2)*C(rightCountX, 2)
            ways_5 = nCr(leftCountX, 2)*nCr(rightCountX, 2) % mod
            
            # freq(x)=4 => among i1,i2,i4,i5, 3 of them are x, 1 is not x
            # We must pick exactly 2 from left side, 2 from right side.
            # Lx+Rx=3 => possibilities: (Lx,Rx)=(2,1) or (1,2).
            # If Lx=2 => pick 2 x from left side => C(leftCountX,2) ways
            #           pick 1 x from right side => C(rightCountX,1) ways
            #           pick 1 not-x from the same side => for the side with 1 x, we must pick 1 not-x,
            #           so if Rx=1 => we also pick 1 not-x from right => (R - rightCountX) ways
            # Similarly for Lx=1 => pick 1 x from left => C(leftCountX,1),
            #                       plus 1 not-x from left => (L - leftCountX),
            #                       pick 2 x from right => C(rightCountX,2).
            ways_4 = 0
            # Lx=2, Rx=1 => Ln=0, Rn=1
            term1 = nCr(leftCountX, 2) * nCr(rightCountX, 1) % mod
            term1 = term1 * ((R - rightCountX) % mod) % mod
            # Lx=1, Rx=2 => Ln=1, Rn=0
            term2 = nCr(leftCountX, 1) * nCr(rightCountX, 2) % mod
            term2 = term2 * ((L - leftCountX) % mod) % mod
            ways_4 = (term1 + term2) % mod
            
            # freq(x)=3 => among i1,i2,i4,i5, exactly 2 are x
            # We pick 2 from left side, 2 from right side => Lx+Rx=2
            # The other 2 picks are not x (but can be same or different, freq(not-x) <=2 <3 => no tie).
            # possibilities (Lx,Rx) in { (0,2), (1,1), (2,0) }.
            # Then the not-x picks are (2-Lx) from left side, (2-Rx) from right side, no restrictions on duplication
            ways_3 = 0
            # (Lx,Rx) = (0,2)
            tmp = nCr(leftCountX,0)*nCr(rightCountX,2) % mod
            tmp = tmp * nCr(L - leftCountX, 2) % mod
            tmp = tmp * nCr(R - rightCountX, 0) % mod
            ways_3 = (ways_3 + tmp) % mod
            # (Lx,Rx) = (1,1)
            tmp = nCr(leftCountX,1)*nCr(rightCountX,1) % mod
            tmp = tmp * nCr(L - leftCountX, 1) % mod
            tmp = tmp * nCr(R - rightCountX, 1) % mod
            ways_3 = (ways_3 + tmp) % mod
            # (Lx,Rx) = (2,0)
            tmp = nCr(leftCountX,2)*nCr(rightCountX,0) % mod
            tmp = tmp * nCr(L - leftCountX, 0) % mod
            tmp = tmp * nCr(R - rightCountX, 2) % mod
            ways_3 = (ways_3 + tmp) % mod
            
            # freq(x)=2 => the trickiest, we use the specialized O(L^2 + R^2) method
            ways_2 = count_freq2_for_j(j)
            
            ways_for_j = (ways_2 + ways_3 + ways_4 + ways_5) % mod
            ans = (ans + ways_for_j) % mod
        
        return ans