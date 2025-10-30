class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        """
        We have an infinite array big_nums built by concatenating the "powerful arrays" of all positive integers:
          - The 'powerful array' for x is the sorted list of powers of two that sum to x (i.e. the bits of x).
          - Example: x=11 -> binary 1011 -> bits at positions {0,1,3} -> powers of two = [1,2,8].
        
        big_nums is then [1,      # for x=1
                          2,      # for x=2
                          1,2,    # for x=3
                          4,      # for x=4
                          1,4,    # for x=5
                          2,4,    # for x=6
                          1,2,4,  # for x=7
                          8,      # for x=8
                          1,8,    # for x=9
                          2,8,    # for x=10
                          1,2,8,  # for x=11
                          4,8,    # for x=12
                          ...
                         ]
        
        The index of big_nums is 0-based.  For example:
         big_nums[0]=1 (from x=1),
         big_nums[1]=2 (from x=2),
         big_nums[2..3]=[1,2] (from x=3),
         big_nums[4]=4 (from x=4), and so on.
        
        Let c(n) = sum of popcount(k) for k=1..n, so c(n) is the total length of big_nums up through integer n.
          That means big_nums[ c(n-1) .. c(n)-1 ] are exactly the powers-of-two array for the integer n.
          (c(0)=0 by convention.)

        We are given queries of the form [from_i, to_i, mod_i], and must compute:
            (big_nums[from_i] * big_nums[from_i+1] * ... * big_nums[to_i]) mod mod_i
        
        Because from_i, to_i can be as large as 10^15, we cannot build the array explicitly.
        Instead, we must compute the product in O(log(...)) time by:
          1) Determining which integer covers index 'from_i' and which covers index 'to_i'.
          2) Partially covering the first integer's sub-array if 'from_i' is not at the start.
          3) Partially covering the last integer's sub-array if 'to_i' is not at the end.
          4) Quickly covering any full ranges of consecutive integers in between by using
             the fact that the full product for an integer x's "powerful array" is 2^(sum_of_bit_positions_of_x).
             And the product over a range of integers [a..b] can be written as
                 2^( sum_{k=a..b} sum_of_positions_of_set_bits_in(k) ).
        
        The main tasks:
          - c(n): sum_{k=1..n} popcount(k).
          - sumpos(n): sum_{k=1..n} (sum of bit positions in k).
          - We do fast exponentiation of 2^E mod M.  But M may not be prime and can be up to 10^5.
            - If M is even, enough factors of 2 can drive the product to 0 mod M.
            - Otherwise, we can use φ(M) or Carmichael's λ(M). In a unified way, we factor out powers of 2
              and apply CRT for the odd part.  This is encapsulated in a helper pow2_mod(...) function.

        Below is a complete, self-contained solution.
        """

        import sys
        sys.setrecursionlimit(10**7)
        
        # -----------------------------
        # Precompute base values for c(2^m - 1) and sumpos(2^m - 1).
        # c(2^m - 1) = m * 2^(m-1)
        # sumpos(2^m - 1) = 2^(m-1) * (m*(m-1)/2),  where bit positions start at 0 for the least significant bit.
        MAX_BITS = 60  # enough for numbers up to around 10^18
        baseC = [0]*(MAX_BITS+1)
        baseSumPos = [0]*(MAX_BITS+1)
        for m in range(1, MAX_BITS+1):
            baseC[m] = m * (1 << (m-1))  # m * 2^(m-1)
            # sumpos(2^m - 1):
            # each bit position k in [0..m-1] occurs 2^(m-1) times among numbers [0..2^m-1),
            # each contribution is k per set, sum(k=0..m-1) k = m*(m-1)/2
            # => 2^(m-1) * (m*(m-1)/2)
            baseSumPos[m] = (1 << (m-1)) * (m*(m-1)//2)

        # Memo dicts for c(n) and sumpos(n).  c(0)=0, sumpos(0)=0.
        cMemo = {0:0}
        sMemo = {0:0}
        
        def sumPopCountUpTo(n: int) -> int:
            """
            c(n) = sum_{k=1..n} popcount(k).
            Recursive formula:
              if n=0 => 0
              let p = 2^m <= n < 2^(m+1), r = n - p
              c(n) = c(p-1) + (r+1) + c(r)
              where c(p-1) = m * 2^(m-1).
            """
            if n in cMemo:
                return cMemo[n]
            if n==0:
                return 0
            # highest power of 2 <= n
            m = n.bit_length()-1
            p = 1 << m  # 2^m
            if p == n+1:
                # means n is 2^m - 1
                cMemo[n] = baseC[m]
                return cMemo[n]
            # otherwise n >= p
            r = n - p
            val = baseC[m] + (r+1) + sumPopCountUpTo(r)
            cMemo[n] = val
            return val

        def sumPosUpTo(n: int) -> int:
            """
            sumpos(n) = sum_{k=1..n} (sum of bit positions of k).
            Let p = 2^m <= n < 2^(m+1), r = n-p.
            sumpos(n) = sumpos(p-1) + sumpos(r) + (r+1)*m
            where sumpos(2^m - 1) = 2^(m-1) * (m*(m-1)/2).
            """
            if n in sMemo:
                return sMemo[n]
            if n==0:
                return 0
            m = n.bit_length()-1
            p = 1 << m
            if p == n+1:
                # n==2^m - 1
                sMemo[n] = baseSumPos[m]
                return sMemo[n]
            r = n - p
            val = baseSumPos[m] + sumPosUpTo(r) + (r+1)*m
            sMemo[n] = val
            return val

        # Find the smallest integer x >= 1 such that c(x) > idx.
        # i.e. big_nums[idx] is in x's "powerful array".
        def findIntegerCoveringIndex(idx: int) -> int:
            # c(x) is non-decreasing. We want the smallest x s.t. c(x) > idx.
            # We'll do a binary search in [1..(some large limit)].
            # Because c(n) ~ average_popcount * n, we won't need to go beyond ~2*(idx).
            # But to be safe for extremes, we can go up to 2^50 or so.
            low, high = 1, (1 << 50)  # plenty large for c(...) up to >10^15
            while low < high:
                mid = (low + high)//2
                if sumPopCountUpTo(mid) > idx:
                    high = mid
                else:
                    low = mid+1
            return low

        # For a single integer x, the product of its entire 'powerful array' is:
        #    2^( sum of set-bit-positions in x ).
        # We'll need that exponent to do the multiplication mod M.
        def sumBitPositions(x: int) -> int:
            # Sum of bit positions where x has a 1.  (Rightmost bit = position 0)
            # Up to ~ 50 bits, so a simple loop is fine.
            pos = 0
            s  = 0
            while x > 0:
                if x & 1:
                    s += pos
                pos += 1
                x >>= 1
            return s

        # Partial coverage in x's powerful array from offsetStart..offsetEnd (0-based within x's bit-list)
        # We'll sum those bit positions, then do 2^that_sum mod M.
        def partialCoverage(x: int, offsetStart: int, offsetEnd: int, modInfo) -> int:
            # Collect the positions of set bits of x (ascending).
            # Then multiply bits[offsetStart..offsetEnd].
            # The product is 2^(sum of those bit positions).
            # offsetEnd is inclusive.
            bits = []
            pos  = 0
            tmp  = x
            while tmp > 0:
                if (tmp & 1) == 1:
                    bits.append(pos)
                pos += 1
                tmp >>= 1
            # sum needed range
            subBits = bits[offsetStart:offsetEnd+1]
            exponent = sum(subBits)
            return pow2_mod(exponent, modInfo)

        # We will do "2^exponent mod m" but m can be up to 10^5, not necessarily prime.
        # If m is even, large exponent can make the product 0.  Use factorization m = 2^a * oddPart,
        # then apply CRT.  We'll precompute everything in getModInfo(m).
        
        from math import gcd

        # Euler's totient (works for the odd part because we only feed it a factor <=10^5).
        def eulerPhi(n: int) -> int:
            # Compute φ(n) by prime factorizing n (<= 100000).
            # That is fast enough with trial division up to sqrt(n).
            phi = n
            x = n
            f = 2
            while f*f <= x:
                if x % f == 0:
                    while x % f == 0:
                        x //= f
                    phi -= phi//f
                f += 1 if f==2 else 2
            if x>1:
                phi -= phi//x
            return phi

        # Extended Euclidean for modular inverse (when gcd is 1).
        def extendedGCD(a, b):
            if b == 0:
                return (1, 0)
            x2, y2 = extendedGCD(b, a%b)
            return (y2, x2 - (a//b)*y2)

        # Precompute mod -> (a, oddPart, inv2a, phiOdd) in a cache.
        modCache = {}

        def getModInfo(m: int):
            if m in modCache:
                return modCache[m]
            # factor out powers of 2
            temp = m
            a = 0
            while temp%2==0:
                temp//=2
                a+=1
            oddPart = temp  # remaining part is odd
            # If oddPart>1, compute phi(oddPart), else if oddPart=1 => everything is a power of 2
            if oddPart>1:
                phiOdd = eulerPhi(oddPart)
            else:
                phiOdd = 1  # doesn't matter, we won't use it if oddPart=1

            # If oddPart>1, we also want inv(2^a mod oddPart) for CRT.  If a=0, it's trivial (1).
            # If oddPart=1, we won't need CRT actually (the mod is purely 2^a).
            if oddPart>1 and a>0:
                # 2^a mod oddPart might be invertible because gcd(2^a, oddPart)=1 (oddPart is odd).
                twoA = (1<<a) % oddPart
                # inverse via exponent or extended Euclid
                # gcd(twoA, oddPart)=1 => we can do extendedGCD
                inv, _ = extendedGCD(twoA, oddPart)
                inv2a = (inv % oddPart)
            else:
                inv2a = 1

            modCache[m] = (a, oddPart, inv2a, phiOdd)
            return modCache[m]

        def pow2_mod(exponent: int, modInfo) -> int:
            """
            Compute (2^exponent) mod m using the factorization-based approach.
            modInfo = (a, oddPart, inv2a, phiOdd).
            """
            (a, oddPart, inv2a, phiOdd) = modInfo
            m = (1<<a)*oddPart  # reconstruct

            # If exponent >= a, then factor 2^a divides out, leaving at least 2^(exponent-a) factor.
            # That means the result is 0 if a>0.  (Because 2^a | 2^exponent, so mod 2^a is 0,
            # and thus mod m is also 0 whenever exponent >= a.)
            if a>0 and exponent>=a:
                return 0

            # otherwise exponent < a => the 2^exponent mod 2^a is simply (1<<exponent).
            part2a = (1<<exponent)  # guaranteed exponent < a < 60, safe in Python
            if oddPart==1:
                # then m=2^a, so the result mod 2^a is just part2a
                return part2a % m

            # For the odd part, we reduce exponent mod phiOdd (since gcd(2, oddPart)=1):
            exponentMod = exponent % phiOdd
            partOdd = pow(2, exponentMod, oddPart)

            # Combine via CRT:
            # we want a number Y such that:
            #   Y ≡ part2a (mod 2^a)
            #   Y ≡ partOdd (mod oddPart)
            # Let Y = part2a + k*(2^a). We want (part2a + k*(2^a)) ≡ partOdd (mod oddPart).
            # => k*(2^a) ≡ (partOdd - part2a) (mod oddPart).
            diff = (partOdd - part2a) % oddPart
            # multiply by inverse of 2^a mod oddPart, which is inv2a
            k = (diff * inv2a) % oddPart
            Y = part2a + ((1<<a)*k)
            return Y % m

        # Multiply two numbers mod m, just (x*y)%m.  Python handles large int fine.
        def mul_mod(a, b, m): 
            return (a*b) % m

        answers = []

        for (startIdx, endIdx, mod) in queries:
            # Handle degenerate case
            if startIdx > endIdx:
                # not really expected by the problem statement, but just in case
                answers.append(1 % mod)
                continue

            modInfo = getModInfo(mod)
            
            # 1) Find the integer covering 'startIdx' => call it L
            L = findIntegerCoveringIndex(startIdx)
            offsetL = startIdx - sumPopCountUpTo(L-1) if L>1 else startIdx  # offset within L's array
            
            # 2) Find the integer covering 'endIdx' => call it R
            R = findIntegerCoveringIndex(endIdx)
            offsetR = endIdx - sumPopCountUpTo(R-1) if R>1 else endIdx

            productMod = 1

            if L == R:
                # Entire subrange [startIdx..endIdx] lies within the same integer's powerful array.
                productMod = partialCoverage(L, offsetL, offsetR, modInfo)
            else:
                # partial coverage for L from offsetL..end of L's array
                fullLenL = (bin(L).count('1'))
                if offsetL <= fullLenL -1:
                    productMod = mul_mod(productMod,
                                         partialCoverage(L, offsetL, fullLenL-1, modInfo),
                                         mod)

                # now skip integers from L+1..R-1 fully (if L+1 <= R-1)
                if L+1 <= R-1:
                    # The product over all x in [L+1..R-1] = 2^( sumpos(R-1) - sumpos(L) ).
                    # Because the entire powerful array of x contributes exponent sumBitPositions(x).
                    expBlock = sumPosUpTo(R-1) - sumPosUpTo(L)
                    blockVal = pow2_mod(expBlock, modInfo)
                    productMod = mul_mod(productMod, blockVal, mod)

                # partial coverage for R from offset 0..offsetR
                if offsetR >= 0:
                    productMod = mul_mod(productMod,
                                         partialCoverage(R, 0, offsetR, modInfo),
                                         mod)

            answers.append(productMod % mod)

        return answers