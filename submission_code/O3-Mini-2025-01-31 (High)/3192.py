class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        mod = 10**9 + 7
        # P = 2^n
        P = 1 << n
        # Get lower n bits and higher bits of a and b.
        A0 = a & (P - 1)
        B0 = b & (P - 1)
        A1 = a >> n
        B1 = b >> n
        # Let d be the XOR of the lower parts.
        d = A0 ^ B0
        # Set C = A1, D = B1 for brevity.
        C, D = A1, B1
        # Define A = P*(C+D)+ d   and   F = D*P.
        A_val = P * (C + D) + d
        F_val = D * P

        # We “split” X into two parts:
        #   X = (B) + (Y)
        # where B comes from bit–positions where d=1 and Y from positions where d=0.
        # The maximum possible Y (taking all bits “on” where d is 0) is:
        Y_opt = (P - 1) - d
        # (term1 comes from Y^2 + A*Y, with A = P*(C+D)+ d).
        term1 = Y_opt * Y_opt + A_val * Y_opt

        # Next, we wish to choose B (which must be a sub–mask of d)
        # so that the quadratic
        #    g(B) = (P*(C - D) + d) * B  -  B^2
        # is maximized. (Note that one may check that
        #    P*(C - D) + d = A_val - 2*F_val.)
        T_coeff = P * (C - D) + d
        # Our goal is then to compute
        #    best_B = argmax{ g(B) }    for B allowed (B is a sub–mask of d)
        # If d == 0 then the only allowed value is 0.
        if d == 0:
            best_g = 0
        else:
            # If d equals all ones (i.e. d == P-1) then every B from 0 to P-1 is allowed.
            if d == P - 1:
                # The quadratic g(B)= T_coeff*B - B^2 is concave so its maximum occurs (in reals)
                # at B = T_coeff/2. We try floor and ceil.
                t_target = T_coeff / 2.0
                candidate1 = int(t_target)  # floor
                candidate2 = candidate1 + 1
                # Clamp the candidates to [0, P-1]
                candidate1 = max(0, min(candidate1, P - 1))
                candidate2 = max(0, min(candidate2, P - 1))
                g1 = T_coeff * candidate1 - candidate1 * candidate1
                g2 = T_coeff * candidate2 - candidate2 * candidate2
                best_g = g1 if g1 >= g2 else g2
            else:
                # Otherwise the allowed B's are exactly those numbers that can be formed
                # using only the bit–positions where d has a 1.
                # (In other words, B must be a sub–mask of d.)
                # We list those bit–positions (in ascending order) so that every allowed B is:
                #    B = sum_{j=0}^{k-1} b_j * 2^(pos_list[j])
                pos_list = [i for i in range(n) if (d >> i) & 1]
                k = len(pos_list)
                # Define a function that, given an integer r (0 ≤ r < 2^k),
                # returns the sub–mask corresponding to its binary expansion:
                def get_submask(r: int) -> int:
                    s = 0
                    j = 0
                    while r:
                        if r & 1:
                            s += (1 << pos_list[j])
                        r //= 2
                        j += 1
                    return s
                # Since the mapping r -> get_submask(r) is monotonic (the allowed B's are
                # exactly the sub–masks of d listed in increasing order) we can binary search
                # for the allowed B that is closest to t_target = T_coeff/2.
                t_target = T_coeff / 2.0
                lo = 0
                hi = (1 << k) - 1
                ans_r = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    # Compute the candidate submask
                    val = 0
                    for j in range(k):
                        if (mid >> j) & 1:
                            val += (1 << pos_list[j])
                    if val <= t_target:
                        ans_r = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                candidate_vals = []
                r1 = ans_r
                val1 = 0
                for j in range(k):
                    if (r1 >> j) & 1:
                        val1 += (1 << pos_list[j])
                candidate_vals.append(val1)
                if ans_r + 1 < (1 << k):
                    r2 = ans_r + 1
                    val2 = 0
                    for j in range(k):
                        if (r2 >> j) & 1:
                            val2 += (1 << pos_list[j])
                    candidate_vals.append(val2)
                if ans_r - 1 >= 0:
                    r0 = ans_r - 1
                    val0 = 0
                    for j in range(k):
                        if (r0 >> j) & 1:
                            val0 += (1 << pos_list[j])
                    candidate_vals.append(val0)
                best_g = None
                for B_val in candidate_vals:
                    g_val = T_coeff * B_val - B_val * B_val
                    if best_g is None or g_val > best_g:
                        best_g = g_val
        term2 = best_g
        # The “constant” – coming from the high–parts – is
        term3 = D * P * (C * P + d)
        # Hence the maximum product is
        f_opt = term1 + term2 + term3
        return f_opt % mod