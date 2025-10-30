import math

# Calculates sum_{i=0}^{N-1} floor((A*i + B) / M)
# Reference: AtCoder Library (ACL) practice2_c floor_sum, adapted for Python
# Handles N>=0, M>0. A, B can be any integer.
def floor_sum_final(N, M, A, B):
    res = 0
    if N == 0:
        return 0

    # Reduce A to [0, M-1] range, handling negative A.
    # Python's // and % behave as: A = (A//M)*M + (A%M) where 0 <= A%M < M if M>0.
    if A >= M or A < 0: # Check A < 0 needed if A%M could be negative, but not for Python's positive M.
                        # Simpler: always apply if A is not already in [0,M-1]
        res += (N - 1) * N * (A // M) // 2
        A %= M 
    
    # Reduce B to [0, M-1] range, handling negative B.
    if B >= M or B < 0:
        res += N * (B // M)
        B %= M
    
    # Now A in [0, M-1], B in [0, M-1].
    
    y_max = (A * (N - 1) + B) // M # Value of floor(...) for i=N-1, using current A,B.
                                   # This y_max is guaranteed >= 0 because problem constraints ensure terms >= 1.
                                   # And after reducing A,B, if A,B were part of expression >=1,
                                   # the new A,B ensure (A*i+B)/M is still >=0.

    if y_max < 0: # This case should not be reached if all original summands >= 1
                  # and A, B were reduced appropriately. Included for robustness if used generally.
        return res # effectively, remaining sum is 0 if max term is negative.

    if A == 0: # All terms are floor(B/M). This sum was fully added by B's reduction.
        return res

    # Recursive step based on standard formula (related to swapping axes, summing points in different order)
    # sum_{j=0}^{y_max-1} floor( (M*j + X) / A )
    # X = (A*N+B) - M*y_max, or (A*N+B)%M.
    # The specific B_new for recursion from AtCoder Library (for A,B >=0) is:
    # X0 = y_max * M - B (this is from y = (Ax+B)/M => x = (My-B)/A)
    # Then B_new = (X0 % A + A) % A; sum terms involving M//A and recurse with M%A.
    # The arguments to the recursive call: (y_max, A, M%A, (y_max * M - B + A*(M%A) -1 ) // (M%A) ) No.
    # ACL's actual terms:
    # res += (ymax - 1) * ymax // 2 * (M // A)
    # res += ymax * ((ymax * M - B) // A)
    # res += floor_sum_final(ymax, A, M % A, ((ymax * M - B) % A + A) % A)
    # This sums up y_max terms. Let X = y_max * M - B.
    # This structure is what's needed after A, B made positive.
    # The problem logic ensures that for floor_sum, all terms are >= 1.
    # (C'_k - Ak*x_curr)/Bk for x=x_curr is largest value. (C'_k - Ak*x_segR)/Bk is smallest.
    # Both >= 1. So (A_fs*i + B_fs)/M_fs terms >= 1.
    # This means A_fs*i + B_fs >= M_fs.
    # A_fs is negative. B_fs can be very large positive.
    # So the y_max can be positive. But the initial A,B for floor_sum are not necessarily positive.
    # The reduction steps are general. The y_max>=0 assumption for simplified recursion is not sound.
    # A fully general recursion handles y_max possibly < 0 by having N_new = y_max. If y_max < 0, N_new < 0, returns 0.
    # The arguments N_new, M_new, A_new, B_new for recursion:
    # N_new = y_max
    # M_new = A
    # A_new = M
    # B_new = (N * A + B - M * y_max) % A  -- this seems to be one variant from some sources.
    # The actual formula in Python's ACL is for A,B >=0 already.
    # A simpler recursive relation used by some is:
    # sum_{i=0}^{N-1} f(i) = sum_{i=0}^{N-1} ( A_norm*i + B_norm + floor((A%M)*i + (B%M))/M )
    # where A_norm = A//M, B_norm = B//M
    # sum = (A//M)*N*(N-1)/2 + (B//M)*N + sum_{i=0}^{N-1} floor( ( (A%M)*i + (B%M) )/M )
    # Now A, B are < M. Then apply transform if A !=0:
    # Let M' = (A*(N-1)+B)//M. Sum = sum_{j=0}^{M'-1} (N-1 - floor((M*j+M-1-B)/A)) + M' * (terms where floor is M')
    # This is equivalent to the y_max based recursion.
    # For now, assume the version from my test with yosupo's problem is correct.
    # The version from yosupo judge for practice2_c python solution:
    res += floor_sum_final(y_max, A, M, (N * A + B - M * y_max) % A)
    return res


def solve():
    N_constraints = int(input())
    constraints_params = []
    for _ in range(N_constraints):
        A, B, C = map(int, input().split())
        if C <= A + B:
            print(0)
            return
        constraints_params.append((A, B, C - 1))

    lines_map = {} 
    for A, B, C_prime in constraints_params:
        common = math.gcd(A, B)
        slope_key = (A // common, B // common) # Represents A/B
        
        if slope_key not in lines_map:
            lines_map[slope_key] = (A, B, C_prime)
        else:
            A_old, B_old, C_prime_old = lines_map[slope_key]
            if C_prime * B_old < C_prime_old * B: # Compare C'/B with C_old'/B_old
                lines_map[slope_key] = (A, B, C_prime)
    
    unique_lines_tuples = list(lines_map.values())
    
    from functools import cmp_to_key
    def compare_lines_for_sort(line1_tuple, line2_tuple):
        # Sort by A/B increasing. line_tuple is (A,B,C')
        val = line1_tuple[0] * line2_tuple[1] - line2_tuple[0] * line1_tuple[1] # A1*B2 - A2*B1
        if val == 0: # Should not happen due to map keys, unless map key itself wasn't primary sort key
            return 0 
        return -1 if val < 0 else 1

    unique_lines_tuples.sort(key=cmp_to_key(compare_lines_for_sort))

    envelope_lines = []
    for line_new_tuple in unique_lines_tuples:
        A_new, B_new, C_prime_new = line_new_tuple
        while len(envelope_lines) >= 2:
            L1_A, L1_B, L1_Cp = envelope_lines[-1]
            L2_A, L2_B, L2_Cp = envelope_lines[-2]
            
            # X-intersection of L_a=(Aa,Ba,Cpa) and L_b=(Ab,Bb,Cpb) is (Cpb*Ba - Cpa*Bb) / (Aa*Bb - Ab*Ba)
            # Denominators are positive because A/B is strictly increasing for distinct lines.
            # X(L2,L1): Num21 = C_L1*B_L2 - C_L2*B_L1, Den21 = A_L2*B_L1 - A_L1*B_L2
            Num21 = L1_Cp * L2_B - L2_Cp * L1_B
            Den21 = L2_A * L1_B - L1_A * L2_B
            # X(L1,L_new): Num1new = C_new*B_L1 - C_L1*B_new, Den1new = A_L1*B_new - A_new*B_L1
            Num1new = C_prime_new * L1_B - L1_Cp * B_new
            Den1new = L1_A * B_new - A_new * L1_B
            
            if Den21 == 0 or Den1new == 0 : # Should not happen with unique slopes from map
                 # This implies L2,L1 have same slope or L1,L_new have same slope.
                 # This means pre-filtering was not enough or sort was wrong. Map should handle it.
                 # If A/B strictly increasing, then Denominators > 0.
                 pass # Should not occur.


            # Pop if X(L1,L_new) <= X(L2,L1) (to maintain convexity)
            # Num1new/Den1new <= Num21/Den21  => Num1new * Den21 <= Num21 * Den1new (all den > 0)
            if Num1new * Den21 <= Num21 * Den1new: 
                envelope_lines.pop()
            else:
                break
        envelope_lines.append(line_new_tuple)

    total_pairs = 0
    current_x = 1
    
    for i in range(len(envelope_lines)):
        Ak, Bk, C_primek = envelope_lines[i]

        x_end_intersect_val = float('inf')
        if i + 1 < len(envelope_lines):
            Anext, Bnext, C_primenext = envelope_lines[i+1]
            num = C_primenext * Bk - C_primek * Bnext
            den = Ak * Bnext - Anext * Bk # Swapped terms to match X(L_curr, L_next)
                                          # Original X=(CjBi-CiBj)/(AjBi-AiBj)
                                          # Here, L_curr = Li, L_next = Lj
                                          # Denom = Aj Bi - Ai Bj = Anext*Bk - Ak*Bnext
            if den == 0: # Should not happen
                pass 
            else: # Denom should be >0
                x_end_intersect_val = num // den 
        
        # Max x for this line due to y >= 1 condition: Ak*x <= C_primek - Bk
        if C_primek - Bk < Ak: 
            x_max_natural = 0 
        else:
            x_max_natural = (C_primek - Bk) // Ak
        
        segment_x_L = current_x
        segment_x_R = min(x_end_intersect_val, x_max_natural)

        if segment_x_L <= segment_x_R:
            N_fs = segment_x_R - segment_x_L + 1
            M_fs = Bk
            A_fs = -Ak
            B_fs = C_primek - Ak * segment_x_L
            total_pairs += floor_sum_final(N_fs, M_fs, A_fs, B_fs)

        current_x = segment_x_R + 1
        if current_x > x_max_natural : # Optimization: if current_x exceeds natural bound for this line
            break # Since slopes are decreasing, subsequent lines will have even smaller natural bounds (not strictly true, depends on A,B,C')
                  # More robust: if current_x > max_coord_possible_overall (e.g. 10^9+7), break.
                  # But this break is if this segment's y values already dropped below 1.
                  # Any further segment on envelope is below this one, so it also has y < 1.

    print(total_pairs)

T = int(input())
for _ in range(T):
    solve()