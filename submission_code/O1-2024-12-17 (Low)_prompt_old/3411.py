class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        """
        We have an infinite array big_nums formed by concatenating, in ascending order of i=1,2,3,...,
        the "powerful array" of i. The powerful array of i is the list of powers-of-two (in ascending order)
        that sum to i (i.e. the set bits of i in binary form). For example:
          i = 11 (binary 1011) -> [1, 2, 8].
        
        Hence, big_nums looks like:
          i=1 -> [1]
          i=2 -> [2]
          i=3 -> [1,2]
          i=4 -> [4]
          i=5 -> [1,4]
          i=6 -> [2,4]
          i=7 -> [1,2,4]
          i=8 -> [8]
          ...
        Concatenating:
          big_nums = [1, 2, 1,2, 4, 1,4, 2,4, 1,2,4,8, ...]
                     ^   ^^^^^   ^^^   ^^^^   etc.
                     i=1 i=2..3  i=4   i=5..6  i=7..8  ...
        
        We use 0-based indexing on big_nums. Thus:
          big_nums[0] = 1  (chunk for i=1)
          big_nums[1] = 2  (chunk for i=2)
          big_nums[2] = 1  (chunk for i=3)
          big_nums[3] = 2  (chunk for i=3)
          big_nums[4] = 4  (chunk for i=4)
          big_nums[5] = 1  (chunk for i=5)
          ...
        
        We define:
          S(i) = total length of big_nums up through (and including) the chunk for i.
               = sum_{k=1..i} (popcount(k))
          so chunk i occupies indices [S(i-1), S(i)-1] in big_nums, with S(0)=0 by convention.
        
        Also define:
          T(i) = sum of "sum_of_bit_positions(k)" for k=1..i
               where "sum_of_bit_positions(k)" = sum of the positions of set bits in k
               (e.g. for k=11(b1011), the set bits are at positions 0,1,3 -> sum=0+1+3=4).
          The product of the entire chunk for i (the powers of two that make up i) is 2^(sum_of_bit_positions(i)).
        
        For a query [from_idx, to_idx, mod], we want:
          product( big_nums[from_idx] * ... * big_nums[to_idx] ) % mod.
        
        Observing that if a query spans entire chunks from i0+1..i1-1 fully, the product of those
        full chunks is:
          ∏ (2^(sum_of_bit_positions(k))) for k in [i0+1..i1-1]
          =  2^(∑(sum_of_bit_positions(k)) for k in [i0+1..i1-1])
          =  2^( T(i1-1) - T(i0) ).
        
        We only need to do partial-chunk multiplication for the chunk containing 'from_idx' and
        the chunk containing 'to_idx' (if they're not the same chunk).
        
        Steps to solve each query efficiently:
          1) Find which chunk (integer) i0 contains index from_idx.
          2) Find which chunk (integer) i1 contains index to_idx.
          3) If i0 == i1, just multiply the slice within that chunk. 
             Otherwise, multiply:
                - partial slice in chunk i0 from (from_idx - S(i0-1)) .. end,
                - all full chunks from i0+1 .. i1-1 (if any),
                - partial slice in chunk i1 from beginning .. (to_idx - S(i1-1)).
        
        Each of these products is taken modulo 'mod' for that query.
        
        To implement the above, we need two main pieces quickly for large i (up to around 10^15 or more of big_nums indices):
          - Invert the function S(i) = sum_{k=1..i} popcount(k), i.e. quickly find chunk i given an index x in big_nums.
          - Quickly compute T(i) = sum_{k=1..i} sum_of_bit_positions(k).
          Both can be computed using a "highest-bit decomposition" or digit-DP in O(log n).
        
        Outline:
          - We will implement two functions:
              popcount_prefix(n) -> sum_{k=1..n} popcount(k).
              bitpossum_prefix(n) -> sum_{k=1..n} sum_of_bit_positions(k).
            Each can be computed in O(log(n)) through a known highest-bit decomposition.
            
          - Then to invert popcount_prefix (i.e. find i s.t. popcount_prefix(i-1) <= x < popcount_prefix(i)):
            we can do a simple binary search in [0..some_upper], using our popcount_prefix function
            to check mid. The maximum needed for i is at most about 1.5*10^15 or so (since sum of popcounts
            grows roughly ~ number_of_bits * n/2, but we take a safe upper bound, e.g. 2^50).
            
          - Once we have i0 and i1, we handle partial slices by enumerating the set bits of i0 or i1
            in ascending order. This is at most ~60 bits (if i can be up to 2^60), which is fast enough
            for the partial chunk edges. The full chunk range is handled using the exponent
            T(i_end) - T(i_start-1) in pow(2, exponent, mod).
        
        We'll do this for each query. Because queries.length <= 500, this approach is efficient enough.
        """

        import sys
        sys.setrecursionlimit(10**7)
        
        # -----------------------------
        # 1) Precompute popcount_prefix(n) and bitpossum_prefix(n) using recursion with highest bit
        #    We define "popcount_prefix(n)" = sum of popcount(k) for k=1 to n
        #    We define "bitpossum_prefix(n)"= sum of sum_of_bit_positions(k) for k=1..n
        #
        # We'll implement them as functions that do 0..n, then subtract the 0..0 portion if needed.
        # But for convenience, we'll define them directly 1..n.
        # The standard approach:
        #   Let highest bit of n be b, with 2^b <= n < 2^(b+1),
        #   let r = n - 2^b.
        #
        #   sum_popcount(1..n) = sum_popcount(1..(2^b -1)) + (r+1) + sum_popcount(1..r)
        #        because for all k in [2^b .. n], the highest bit is 1, so that accounts for (r+1) ones,
        #        plus we add the popcounts of the lower bits, which is sum_popcount(1..r).
        #
        #   sum_popcount(1..(2^b -1)) = b * 2^(b-1)
        #       (well-known formula for sum of popcounts up to a power of two minus one).
        #
        #   sum_bitpos(1..n) = sum_bitpos(1..(2^b -1)) + (r+1)*b + sum_bitpos(1..r)
        #       because every k in [2^b..n] has the highest bit b set, so that contributes (r+1)*b,
        #       plus the sum of lower bit positions for those k, which is sum_bitpos(1..r).
        #
        #   sum_bitpos(1..(2^b -1)) = b * (some formula?), we can do a similar recursion or keep in a memo.
        #
        # We'll implement these with recursion + memo for speed.
        #
        #   sum_popcount_p2[b] = sum of popcount(1..(2^b -1)) = b * 2^(b-1).
        #   sum_bitpos_p2[b]  = sum of sum_of_bit_positions(k) for k=1..(2^b -1).
        # We'll precompute sum_bitpos_p2[b] by recursion or direct formula:
        #   sum_bitpos_p2[b] = sum_bitpos(1..(2^b -1)) = 
        #       = sum_bitpos(1..(2^(b-1)-1)) + (2^(b-1))* (b-1) + sum_bitpos(1..(2^(b-1)-1))
        #       doubling logic; but let's just fill it up incrementally for b up to 60.
        
        sys.setrecursionlimit(10**7)

        # Precompute up to 60 bits
        MAXB = 60
        # sum of popcount for 1..(2^b -1)
        popcount_p2 = [0]*(MAXB+1)
        # sum of bit positions for 1..(2^b -1)
        bitpos_p2 = [0]*(MAXB+1)

        # Base case for b=0: 2^0=1, range 1..(1-1)=empty -> sums = 0
        # We'll fill for b >= 1
        for b in range(1, MAXB+1):
            # 2^b
            two_b = 1 << b
            # sum of popcounts up to 2^b - 1 is b * 2^(b-1)
            popcount_p2[b] = b * (1 << (b-1))
        # Now fill bitpos_p2 by recursion:
        # sum_bitpos(1..(2^b -1)) can be broken in half or done iteratively from b=1 upward
        #
        # We'll do: bitpos_p2[b] = bitpos_p2[b-1]
        #      + sum of bitpos in [2^(b-1), 2^b -1].
        # But for k in [2^(b-1), 2^b-1], the highest bit is (b-1), so that contributes (2^(b-1))* (b-1),
        # plus the sum_bitpos(1..(2^(b-1)-1)) (the lower bits).
        # So:
        #   bitpos_p2[b] = bitpos_p2[b-1] + (1<<(b-1))*(b-1) + bitpos_p2[b-1].
        #   = 2*bitpos_p2[b-1] + (1<<(b-1))*(b-1).
        bitpos_p2[0] = 0
        for b in range(1, MAXB+1):
            bitpos_p2[b] = 2 * bitpos_p2[b-1] + (1 << (b-1)) * (b-1)
        
        # Define recursion + memo for popcount_prefix and bitpossum_prefix
        from functools import lru_cache

        @lru_cache(None)
        def popcount_prefix(n: int) -> int:
            """Returns sum_{k=1..n} popcount(k)."""
            if n <= 0:
                return 0
            # find highest set bit
            b = n.bit_length() - 1  # highest bit index
            if b == 0:
                # n <= 1
                return n  # popcount(1)=1, popcount_prefix(1)=1
            # p = 2^b
            p = 1 << b
            r = n - p
            # sum_popcount(1..n) = sum_popcount(1..(2^b-1)) + (r+1) + sum_popcount(1..r)
            return popcount_p2[b] + (r + 1) + popcount_prefix(r)

        @lru_cache(None)
        def bitpossum_prefix(n: int) -> int:
            """Returns sum_{k=1..n} sum_of_bit_positions(k)."""
            if n <= 0:
                return 0
            b = n.bit_length() - 1
            if b == 0:
                # n <= 1
                # sum_of_bit_positions(1) = 0
                return 0
            p = 1 << b
            r = n - p
            # sum_bitpos(1..n) = sum_bitpos(1..(2^b-1)) + (r+1)*b + sum_bitpos(1..r)
            return bitpos_p2[b] + (r+1)*b + bitpossum_prefix(r)

        # Helper to get S(i) = sum_{k=1..i} popcount(k)
        def S(i: int) -> int:
            return popcount_prefix(i)

        # Helper to get T(i) = sum_{k=1..i} sum_of_bit_positions(k)
        def T(i: int) -> int:
            return bitpossum_prefix(i)

        # We want to invert S(i) for an index x, i.e. find smallest i such that S(i) > x.
        # We'll do a standard binary search in a certain range; 
        # indices can go up to 10^15, so let's pick an upper bound big enough.
        # sum_{k=1..n} popcount(k) roughly ~ n * average_bitcount. For n ~ 2e14 or so, 
        # that sum can exceed 10^15. Let's pick an upper bound ~ 2e15 or 2^50 to be safe.
        
        def find_chunk_index(x: int) -> int:
            """
            Returns smallest i such that S(i) > x.
            i.e. big_nums[x] belongs to chunk i if S(i-1) <= x < S(i).
            This function returns i (1-based chunk index).
            
            If x < 0, then i=0 is consistent with S(0)=0, meaning nothing.
            But queries can have from=0.. so x won't be negative in normal usage.
            """
            if x < 0:
                return 1  # edge case, not really used typically
            lo, hi = 0, (1 << 50)
            while lo < hi:
                mid = (lo + hi) >> 1
                if S(mid) > x:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        
        # Function to get the product of partial chunk i from sub_start..sub_end (0-based inside that chunk),
        # mod m. The chunk for i has popcount(i) elements which are the powers of two corresponding to
        # the set bits of i in ascending order. That is: [2^b1, 2^b2, ...].
        # sub_start..sub_end are inclusive 0-based indices into that chunk.
        def partial_chunk_product(i: int, start_idx: int, end_idx: int, m: int) -> int:
            """
            i is the integer whose chunk we are partially using.
            We want the product of big_nums[S(i-1)+start_idx .. S(i-1)+end_idx], inclusive.
            The chunk is the set bits of i in ascending order. So we find those bit positions,
            keep only [start_idx..end_idx], and multiply 2^(bit_position).
            
            Return that product mod m.
            """
            # gather set bits of i from lowest to highest
            # up to 60 bits is feasible
            bit_positions = []
            pos = 0
            val = i
            while val > 0:
                if val & 1:
                    bit_positions.append(pos)
                pos += 1
                val >>= 1
            # product is 2^( sum(bit_positions[start_idx..end_idx]) ) mod m
            exp_sum = sum(bit_positions[start_idx:end_idx+1])
            return pow(2, exp_sum, m)

        # Now we handle each query.
        ans = []
        for from_i, to_i, mod_i in queries:
            # find chunk containing from_i
            i0 = find_chunk_index(from_i)  # 1-based chunk index
            i1 = find_chunk_index(to_i)    # 1-based chunk index

            # If from_i is in chunk i0, that means S(i0-1) <= from_i < S(i0).
            # partialLeft index inside chunk i0:
            start_idx_in_i0 = from_i - S(i0-1)
            end_idx_in_i0 = S(i0) - S(i0-1) - 1  # last index in chunk i0 is popcount(i0)-1

            # Similarly for to_i:
            start_idx_in_i1 = 0
            end_idx_in_i1 = to_i - S(i1-1)

            product_mod = 1
            
            if i0 == i1:
                # entire query is in the same chunk
                product_mod = partial_chunk_product(i0, start_idx_in_i0, end_idx_in_i1, mod_i)
            else:
                # partial chunk i0
                if start_idx_in_i0 <= end_idx_in_i0:
                    part_left = partial_chunk_product(i0, start_idx_in_i0, end_idx_in_i0, mod_i)
                    product_mod = (product_mod * part_left) % mod_i
                
                # full chunks i0+1 .. i1-1
                if (i0 + 1) <= (i1 - 1):
                    # we want product of chunk i for i in [i0+1.. i1-1]
                    # = 2^( sum_of_bit_positions(i0+1.. i1-1) ) mod m
                    # exponent = T(i1-1) - T(i0)
                    exp_val = T(i1-1) - T(i0)
                    full_chunks = pow(2, exp_val, mod_i)
                    product_mod = (product_mod * full_chunks) % mod_i

                # partial chunk i1
                if start_idx_in_i1 <= end_idx_in_i1:
                    part_right = partial_chunk_product(i1, start_idx_in_i1, end_idx_in_i1, mod_i)
                    product_mod = (product_mod * part_right) % mod_i
            
            ans.append(product_mod)
        
        return ans