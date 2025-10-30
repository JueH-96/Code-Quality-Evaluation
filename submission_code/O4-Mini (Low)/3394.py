class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # We need an increasing sequence of n numbers nums[i]>=1,
        # nums[0]=x, each nums[i]&x == x, strictly increasing,
        # minimizing nums[n-1].  Equivalently, we pick offsets t
        # so that nums[i]=x + t_i, t_i>=0, t_i & x == 0, and t_i strictly increasing.
        # The smallest sequence of t_i is just the sorted list of non‐negative
        # integers whose bits are disjoint from x.  In fact t_0=0, t_1=1 in that
        # "filtered" set, etc.  One can show the k-th (0-based) such t is obtained
        # by writing k in binary and mapping its bits onto the zero-bit positions of x.
        #
        # So if k = n-1, then
        #   let zeros = [p0,p1,...] be the bit positions where x has a 0-bit,
        #   and let k's binary digits be b0,b1,... in LSB-first order,
        #   then t_k = sum_{i} b_i << p_i.
        # Answer = x + t_{n-1}.
        
        # k-th offset
        k = n - 1
        # collect zero-bit positions of x, up to enough bits for k
        zeros = []
        bitpos = 0
        # we need at most log2(k)+1 bits
        needed = k.bit_length() if k>0 else 1
        while len(zeros) < needed:
            if not (x >> bitpos) & 1:
                zeros.append(bitpos)
            bitpos += 1
        # build t_k
        t = 0
        i = 0
        kk = k
        while kk:
            if kk & 1:
                t |= 1 << zeros[i]
            kk >>= 1
            i += 1
        return x + t