class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        # If n == 0, x can only be 0.
        if n == 0:
            return (a * b) % MOD

        # M = 2^n, mask for low n bits
        M = 1 << n
        mask = M - 1

        # Split a, b into high and low parts
        P = a & mask        # low n bits of a
        Q = b & mask        # low n bits of b
        A_hi = a >> n       # high part of a
        B_hi = b >> n       # high part of b

        # S_bits: positions where P_i == Q_i
        # R_bits: positions where P_i != Q_i
        S_sum = 0
        R_bits = []
        for i in range(n):
            pi = (P >> i) & 1
            qi = (Q >> i) & 1
            if pi == qi:
                # we will set u_i = v_i = 1 => P_i xor x_i = 1
                # that forces x_i = 1 - pi
                S_sum += (1 << i)
            else:
                R_bits.append(i)
        # Sum of weights in R_bits
        R_sum = 0
        for i in R_bits:
            R_sum += (1 << i)

        # We derived that the objective on the R_bits reduces to
        # maximizing a concave quadratic in a = sum of selected weights.
        # The continuous optimum is at a_target = K/2, where
        #   K = M*(B_hi - A_hi) + R_sum
        K = (B_hi - A_hi) * M + R_sum

        # Greedily pick from largest R_bits downward to best approximate K/2.
        # We include weight w_i if 2*(current + w_i) <= K.
        selected = set()
        a_acc = 0
        for i in sorted(R_bits, reverse=True):
            w = 1 << i
            if 2 * (a_acc + w) <= K:
                a_acc += w
                selected.add(i)

        # Build x bitwise
        x = 0
        for i in range(n):
            pi = (P >> i) & 1
            qi = (Q >> i) & 1
            if pi == qi:
                # bit in S => x_i = 1 - pi
                xi = 1 - pi
            else:
                # bit in R => if selected, assign to u => x_i = 1 - pi
                # otherwise assign to v => x_i = pi
                if i in selected:
                    xi = 1 - pi
                else:
                    xi = pi
            if xi:
                x |= (1 << i)

        # Compute final product
        U = a ^ x
        V = b ^ x
        return (U % MOD) * (V % MOD) % MOD