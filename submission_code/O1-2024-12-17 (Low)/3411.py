class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        """
        We have an infinite array big_nums obtained by concatenating the "powerful arrays"
        (the sorted powers-of-two decomposition) for each positive integer in ascending order.

        Example of the beginning of big_nums (0-based indexing):
         index: 0, 1, 2, 3, 4,  5, 6, 7, 8, 9, ...
         value: 1, 2, 1, 2, 4,  1, 4, 2, 4, 1, ...
         which comes from:
           1 -> [1]
           2 -> [2]
           3 -> [1, 2]
           4 -> [4]
           5 -> [1, 4]
           6 -> [2, 4]
           7 -> [1, 2, 4]
           8 -> [8]
           9 -> [1, 8]
           ...
        
        Each query = [from_i, to_i, mod_i] asks for the product
          big_nums[from_i] * big_nums[from_i+1] * ... * big_nums[to_i]
        modulo mod_i.

        Key Observations:
         1) Each big_nums element is a power of two, so the product is of the form
               2^(exponent_sum).
         2) We only need to find the sum of the exponents of big_nums[from_i..to_i].
         3) The exponent of a power of two 2^p is p (where 1 -> p=0, 2 -> p=1, 4 -> p=2, ...).
         4) Concatenation structure: for each positive integer x, if x's binary representation
            has bits set at positions p1 < p2 < ... < pk (0-based from the LSB),
            then we append powers 2^p1, 2^p2, ..., 2^pk in ascending order of p.

        So if we define an indexing of big_nums starting at 0, we want:
            product from index L to R = 2^( sum_of_exponents_in_range(L, R) ).

        We'll do:
          product_mod = pow(2, sum_of_exponents_in_range(L,R), mod).

        The main challenge is that from_i/to_i can be up to 10^15, so we cannot explicitly build
        big_nums. We need fast lookups of how many elements appear up to a certain index and
        the sum of their exponents.

        Let's define:
          popcount_prefix(n) = sum_{i=1..n} (number of set bits in i)
            -> This tells us how long the concatenation is up to integer n (i.e. the big_nums
               indices covered by 1..n is popcount_prefix(n)).
          exponent_prefix(n) = sum_{i=1..n} (sum of bit-positions set in i)
            -> This is the total exponent sum contributed by integers [1..n].

        Then the big_nums indices for integer (n+1) start at popcount_prefix(n) and go up to
        popcount_prefix(n+1)-1.

        To get sum_of_exponents_in_range(L, R), define a prefix function:
          P(k) = sum of exponents for big_nums[0..(k-1)].
        Then
          sum_of_exponents_in_range(L, R) = P(R+1) - P(L).

        Computation of P(k): (k is a big_nums index, 0-based)
         1) If k == 0, P(k) = 0.
         2) Otherwise, find the largest n such that popcount_prefix(n) <= k.
            Then the next integer is (n+1); we want the first (k - popcount_prefix(n)) set-bits
            (from lower to higher bit) of (n+1).
         3) Then
            P(k) = exponent_prefix(n) + partial_bits_exponent( (n+1), count_bits ),
            where count_bits = k - popcount_prefix(n).

        We implement popcount_prefix(n) and exponent_prefix(n) using known formulas with
        the highest power of two decomposition. Each is O(log n). Then do a binary search for n
        in [0..something] so that popcount_prefix(n) <= k < popcount_prefix(n+1). That also
        takes O(log n * log(range)) which is acceptable for up to 500 queries with n ~ 1e15.

        Finally, product = 2^(exponentSum) mod m = pow(2, exponentSum, m).
        (Python's built-in pow base^exp % mod is efficient even for large exp.)

        Let's implement the solution.
        """

        import sys
        sys.setrecursionlimit(10**7)

        # 1) Precompute sumPopcount(2^p - 1) and sumExp(2^p - 1) via closed forms:
        #    sumPopcount(2^p - 1) = p * 2^(p-1).
        #    sumExp(2^p - 1) = 2^(p-1) * (p * (p-1) / 2).
        # We'll implement a recursive approach to handle arbitrary n.

        # --- sum of popcounts from 1..n ---
        # Formula:
        #   let highestPower = 2^p <= n, remainder = n - 2^p
        #   sumPopcount(n) = p * 2^(p-1) + (remainder + 1) + sumPopcount(remainder)
        # with sumPopcount(0) = 0.

        def sumPopcount(n: int) -> int:
            if n == 0:
                return 0
            # p = highest set bit of n
            p = n.bit_length() - 1  # so that 1<<p <= n < 1<<(p+1)
            highest = 1 << p        # 2^p
            remainder = n - highest
            # sumPopcount(2^p - 1) = p * 2^(p-1)
            return (p * (highest >> 1)) + (remainder + 1) + sumPopcount(remainder)

        # --- sum of exponents from 1..n ---
        # Each integer i contributes sum of bit positions set in i (0-based).
        # Similar decomposition:
        #   let highestPower = 2^p <= n, remainder = n - 2^p
        #   sumExp(n) = sumExp(2^p - 1) + p*(remainder+1) + sumExp(remainder)
        #
        # We use closed form for sumExp(2^p - 1):
        #       = 2^(p-1) * [p*(p-1)/2]
        # sumExp(0) = 0 by definition (no integers).

        def sumExp2PowMinus1(p: int) -> int:
            # sumExp(2^p -1) = 2^(p-1) * (p*(p-1)//2)
            # Make sure p >= 1. For p=0 => 2^0-1=0 => sumExp(0)=0, that pattern still holds if we
            # interpret it carefully, but let's just handle it safely.
            if p <= 0:
                return 0
            return (1 << (p - 1)) * (p * (p - 1) // 2)

        def sumExponent(n: int) -> int:
            if n == 0:
                return 0
            p = n.bit_length() - 1  # highest power of 2 <= n is (1 << p)
            highest = 1 << p
            remainder = n - highest
            return sumExp2PowMinus1(p) + p * (remainder + 1) + sumExponent(remainder)

        # prefixLen(n) = sumPopcount(n), prefixExponent(n) = sumExponent(n).

        # For k >= 1, we find the largest n such that popcount_prefix(n) <= k-1 (because
        # we want all elements up to index k-1). But to simplify, we do:
        #   We want the largest n s.t. sumPopcount(n) <= (k-1) if k>0
        #   Then leftover = (k-1) - sumPopcount(n).
        #   If k=0, prefixExponentOfIndex(0)=0.
        #
        # Actually, it's slightly easier to define a helper:
        #   prefixExponentOfIndex(k) = sum of exponents for big_nums[0..k-1].
        #   If k=0 => 0.
        #   Else find the largest n with sumPopcount(n) <= k-1.
        #   leftover = (k-1) - sumPopcount(n).
        #   Then partial is the sum of exponents of the first `leftover+1` set bits of (n+1).
        #
        # But an equivalent simpler approach is:
        #   We want the largest n such that sumPopcount(n) <= k. Then leftover = k - sumPopcount(n).
        #   prefixExponentOfIndex(k) = sumExponent(n) + sum_of_lowest_bits_exponent(n+1, leftover)
        # Because big_nums indexing from sumPopcount(n) to sumPopcount(n+1)-1 belongs to integer (n+1).
        #
        # We'll define a function to do that so the indexing lines up with 0-based big_nums.

        def find_n_for_index(k: int) -> int:
            """
            Returns the largest n such that sumPopcount(n) <= k.
            We'll binary-search in [0..some large], because k can be up to ~10^15.
            """
            if k < 0:
                return -1  # no n >= 0 can satisfy sumPopcount(n)<= negative
            lo, hi = 0, 10**15 + 5  # a safe upper bound
            # We'll do a standard binary search
            while lo < hi:
                mid = (lo + hi) >> 1
                if sumPopcount(mid) <= k:
                    lo = mid + 1
                else:
                    hi = mid
            return lo - 1  # lo is the first that fails, so lo-1 is largest that passes

        def sum_of_lowest_c_bits_exponent(x: int, c: int) -> int:
            """
            x has at most ~50 bits if x <= 10^15. We want the sum of the exponents (bit positions)
            of the lowest c set bits in x. The bits are from LSB=position0, next=1, etc.
            We'll collect set bit positions in ascending order, then sum the first c of them.
            """
            if c <= 0:
                return 0
            s = 0
            bits_collected = 0
            pos = 0
            xx = x
            while xx > 0 and bits_collected < c:
                if xx & 1:
                    s += pos
                    bits_collected += 1
                xx >>= 1
                pos += 1
            return s

        def prefix_exponent_of_index(k: int) -> int:
            """
            Returns sum of exponents for big_nums[0..(k-1)].
            If k=0 -> 0
            Otherwise:
              find n with sumPopcount(n) <= k-1 < sumPopcount(n+1).
              leftover = (k-1) - sumPopcount(n)
              answer = sumExponent(n) + sum_of_lowest_bits_exponent(n+1, leftover+1)
            But to avoid off-by-one confusion, we can equivalently do:
              find n with sumPopcount(n) <= k < sumPopcount(n+1).
              leftover = k - sumPopcount(n)
              answer = sumExponent(n) + sum_of_lowest_bits_exponent(n+1, leftover)
            We'll use the second version for convenience.
            """
            if k <= 0:
                return 0
            n = find_n_for_index(k - 1)   # sumPopcount(n) <= k-1
            leftover = (k - 1) - sumPopcount(n)
            # sumExp(n) covers entire exponents up to integer n
            # then add leftover+1 bits (lowest leftover+1 bits) from n+1
            return sumExponent(n) + sum_of_lowest_c_bits_exponent(n + 1, leftover + 1)

        # Now for each query [L, R, mod]:
        #   exponentSum = prefix_exponent_of_index(R+1) - prefix_exponent_of_index(L)
        #   answer = pow(2, exponentSum, mod)

        answers = []
        for f, t, m in queries:
            # sum_of_exponents_in_range = P(t+1) - P(f)
            expo_sum = prefix_exponent_of_index(t + 1) - prefix_exponent_of_index(f)
            # product mod = 2^expo_sum mod m
            # Python pow(base, exp, mod) handles large exp efficiently.
            ans_mod = pow(2, expo_sum, m)
            answers.append(ans_mod)

        return answers