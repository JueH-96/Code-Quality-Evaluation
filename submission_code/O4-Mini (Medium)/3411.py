from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # Compute P(n) = sum_{i=1..n} popcount(i)
        # Compute Q(n) = sum_{i=1..n} sum of bit positions of i
        def compute_P_Q(n):
            # returns (P(n), Q(n))
            if n <= 0:
                return 0, 0
            P_sum = 0
            Q_sum = 0
            k = 0
            pow2 = 1
            # iterate bits k while 2^k <= n
            while pow2 <= n:
                cycle = pow2 << 1  # 2^(k+1)
                full_cycles = n // cycle
                rem = n % cycle
                # count of ones in k-th bit from 1..n
                cnt = full_cycles * pow2 + max(0, rem - pow2 + 1)
                P_sum += cnt
                Q_sum += k * cnt
                k += 1
                pow2 <<= 1
            return P_sum, Q_sum

        # find smallest i >= 1 such that P(i) >= pos
        def find_i_for_pos(pos):
            # doubling to find an upper bound hi
            lo = 1
            hi = 1
            # get P(hi)
            P_hi, _ = compute_P_Q(hi)
            while P_hi < pos:
                hi <<= 1
                P_hi, _ = compute_P_Q(hi)
            # binary search in [lo..hi]
            while lo < hi:
                mid = (lo + hi) // 2
                P_mid, _ = compute_P_Q(mid)
                if P_mid < pos:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        # helper to get sorted list of bit positions of x
        def bit_positions(x):
            bits = []
            k = 0
            while (1 << k) <= x:
                if (x >> k) & 1:
                    bits.append(k)
                k += 1
            return bits

        answers = []
        for L, R, m in queries:
            # convert to 1-based positions
            L1 = L + 1
            R1 = R + 1

            # find start and end integers
            i_start = find_i_for_pos(L1)
            i_end = find_i_for_pos(R1)

            # prefix sums P and Q for boundaries
            P_start_minus, Q_start_minus = compute_P_Q(i_start - 1)
            P_end_minus, Q_end_minus = compute_P_Q(i_end - 1)

            # offsets within the powerful array of the boundary integers
            d = L1 - P_start_minus - 1  # 0-based index in i_start's bits
            e = R1 - P_end_minus - 1    # 0-based index in i_end's bits

            S = 0  # total exponent sum

            # case when the range lies within a single integer's block
            if i_start == i_end:
                bits = bit_positions(i_start)
                # sum bits from d..e inclusive
                S += sum(bits[d:e+1])
            else:
                # partial from the start integer: bits d..end
                bits_start = bit_positions(i_start)
                S += sum(bits_start[d:])

                # partial from the end integer: bits 0..e
                bits_end = bit_positions(i_end)
                S += sum(bits_end[:e+1])

                # full integers in between, if any
                if i_end > i_start + 1:
                    # Q(i_end-1) - Q(i_start)
                    _, Q_im1 = compute_P_Q(i_end - 1)
                    _, Q_i0 = compute_P_Q(i_start)
                    S += (Q_im1 - Q_i0)

            # the product is 2^S mod m
            answers.append(pow(2, S, m))

        return answers