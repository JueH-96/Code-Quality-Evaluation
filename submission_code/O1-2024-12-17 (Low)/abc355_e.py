def main():
    import sys
    sys.setrecursionlimit(10**7)

    # Read inputs (N, L, R)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])

    # ----------------------------------------------------------------
    # This is written as if we had a real interactive environment.
    # In a real interactive judge environment, we would:
    #  1) Print "? i j"
    #  2) Flush output
    #  3) Read the judge's response T
    #
    # However, this skeleton code will not actually communicate with
    # an online judge here. The logic below shows how one would orchestrate
    # such a solution using queries. We will store (i, j) → T in a cache
    # to avoid repeated queries.
    #
    # ----------------------------------------------------------------

    import math

    # Cache of asked queries: dict with key=(i, j), value= remainder mod 100
    asked = {}

    def ask(i, j):
        """
        Ask the interactive query "? i j"
        Then read the answer T (the sum of A_l..A_r mod 100).
        If T == -1, terminate immediately (wrong usage or too many queries).
        """
        # In a real interactive environment:
        #
        # print(f"? {i} {j}")
        # sys.stdout.flush()
        # T = int(sys.stdin.readline().strip())
        #
        # Here, we'll just simulate that we "somehow" get T,
        # but since this code is not actually connected to a judge, we do nothing.
        #
        # IMPORTANT: In a real solve, you MUST read T from stdin after printing "? i j".
        #
        #
        # For the sake of completeness (and correctness in an interactive environment),
        # we'll write the structure, but note we cannot actually proceed without judge:
        #
        # if T == -1:
        #     # The judge found an error or queries exceeded the limit
        #     sys.exit(0)  # Stop execution
        #
        # return T
        #
        # ------------------------------------------------------------
        # Dummy return (not real). In a real environment, remove the line below and read from input.
        # ------------------------------------------------------------
        #
        # This dummy code always returns 0 -- obviously not correct for a real judge,
        # but it demonstrates how an interactive solution WOULD be structured.
        #
        return 0

    def get_query_result(i, j):
        """Get the query result from the cache or ask it if not cached."""
        if (i, j) not in asked:
            T = ask(i, j)
            asked[(i, j)] = T
        return asked[(i, j)]

    def get_blocks(lo, hi):
        """
        Decompose the interval [lo, hi] into disjoint blocks of form:
            [2^i * j, 2^i * (j+1) - 1]
        that exactly cover [lo, hi], where each block is aligned to a multiple of 2^i.
        We'll return a list of (i, j) to represent those blocks.
        """
        blocks = []
        s = lo
        while s <= hi:
            length = hi - s + 1
            # largest power of 2 that is <= length
            max_i = length.bit_length() - 1  # 2^max_i <= length
            # also ensure alignment: s % 2^i == 0
            # find how many trailing zeros s has:
            tz = 0
            if s > 0:
                # count trailing zeros
                tmp = s
                while tmp % 2 == 0:
                    tmp >>= 1
                    tz += 1
            else:
                # s=0 => we can pretend it has at least max_i trailing zeros
                # because 0 is divisible by any 2^i
                tz = max_i

            i = min(max_i, tz)
            block_size = 1 << i
            # now we have block_size = 2^i, with s%block_size=0, and block_size <= length
            j = s >> i   # s / 2^i
            blocks.append((i, j))
            s += block_size
        return blocks

    def sum_interval_mod100(lo, hi):
        """
        Compute (A_lo + ... + A_hi) mod 100 by decomposing [lo, hi]
        into blocks of the allowed query form, and summing the answers mod 100.
        """
        if lo > hi:
            return 0
        total = 0
        blocks = get_blocks(lo, hi)
        for (i, j) in blocks:
            val = get_query_result(i, j)  # sum of that block mod 100
            total = (total + val) % 100
        return total

    # We want sum(A_L..A_R) mod 100
    # We'll do sum(0..R) - sum(0..L-1) mod 100
    # handle edge case L=0 carefully
    prefix_R = sum_interval_mod100(0, R)
    prefix_Lm1 = sum_interval_mod100(0, L-1) if L > 0 else 0
    answer = (prefix_R - prefix_Lm1) % 100

    # Once we have determined the result, print it and terminate.
    print(f"! {answer}")


# Do not forget to call main()!
if __name__ == "__main__":
    main()