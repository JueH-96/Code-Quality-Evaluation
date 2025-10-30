import math

def solve():
    N = int(input())

    # integer_sqrt_check function:
    # Returns integer m if n_val = m*m, else -1.
    def integer_sqrt_check(n_val):
        if n_val < 0:
            return -1
        if n_val == 0:
            return 0
        
        if hasattr(math, 'isqrt'): # Use math.isqrt if available (Python 3.8+)
            m_isqrt = math.isqrt(n_val)
            if m_isqrt * m_isqrt == n_val:
                return m_isqrt
            else:
                return -1
        else: # Fallback for older Python versions
            m_cand_float = n_val**0.5
            if m_cand_float > 4 * 10**9: # Avoid large float to int conversion if too big
                                         # Max possible m is ~1.15*10^9 for N=10^18. This check is precautionary.
                # For very large numbers, binary search for sqrt is safer if not using math.isqrt
                # However, for values up to ~10^18, m_cand_float should be reasonably precise.
                pass

            m_cand = int(m_cand_float)
            
            # Check a small window around m_cand due to potential float precision issues.
            # For N up to 10^18, target_div_3 is up to ~1.33 * 10^18.
            # sqrt(target_div_3) is up to ~1.15 * 10^9.
            # Python floats (64-bit double) have ~15-17 decimal digits of precision.
            # 1.15*10^9 has 10 digits. Precision should be good.
            # Checking m_cand-1, m_cand, m_cand+1 is usually sufficient.
            # A window from m_cand-2 to m_cand+2 is very safe.
            for test_m_offset in range(-2, 3): # Covers m_cand-2, ..., m_cand+2
                cur_m = m_cand + test_m_offset
                if cur_m < 0: # m must be non-negative
                    continue
                # cur_m * cur_m can be very large, but Python handles large integers.
                if cur_m * cur_m == n_val:
                    return cur_m
            return -1

    A = 1
    # Loop for A = x - y. A >= 1.
    # We need A^3 < N for y > 0.
    while True:
        # A*A*A can be large, but Python handles large integers.
        # Max A for N=10^18 is 10^6, so A^3 is at most 10^18.
        # Using A_times_A = A*A; val_A_cubed = A_times_A*A to avoid recomputing A*A
        A_times_A = A * A
        if A > N // A_times_A : # This is A > N / A^2, or A^3 > N. (Avoids A^3 directly if N is small)
                                # More robustly, val_A_cubed = A * A * A. if val_A_cubed >= N: break
                                # Check: if A is very large, A*A*A could overflow standard int types,
                                # but not Python's arbitrary precision int.
                                # Let's use A * A * A, it's clearer.
            pass # Fall through to val_A_cubed check

        val_A_cubed = A * A * A
        
        if val_A_cubed >= N:
            # If A^3 >= N, then N/A <= A^2, so B <= A^2.
            # This implies y <= 0 (as shown in derivation).
            # No solution for this A or any larger A.
            break

        if N % A == 0:
            B = N // A # B = N/A = x^2 + xy + y^2
            
            # We derived y = (m-A)/2 where m^2 = (4B - A^2)/3.
            # Check B > A*A (equivalent to N > A^3), guaranteed by loop condition val_A_cubed < N.
            
            target_for_m_squared_times_3 = 4 * B - A_times_A # Using A_times_A = A*A
            
            # target_for_m_squared_times_3 must be positive and divisible by 3.
            # Positivity: B > A^2 => 4B > 4A^2 => 4B - A^2 > 3A^2 >= 3 (since A>=1).
            
            if target_for_m_squared_times_3 % 3 == 0:
                M_sq = target_for_m_squared_times_3 // 3
                m = integer_sqrt_check(M_sq)

                if m != -1: # M_sq is a perfect square m*m
                    # We need y = (m-A)/2.
                    # For y > 0, we need m > A. This is guaranteed by N > A^3.
                    # For y to be an integer, (m-A) must be even.
                    if (m - A) > 0 and (m - A) % 2 == 0: # m > A check is mostly for clarity
                        y = (m - A) // 2
                        x = y + A
                        
                        print(x, y)
                        return # Exit function, solution found
        A += 1
        # Safety break for A. Max A for N=10^18 is 10^6.
        # 2*10^6 is a generous upper bound.
        # The val_A_cubed >= N check is the primary termination for this loop.
        if A > 2 * (10**6) + 5: # Max A for N=10^18 is 10^6. Max A for N=8*10^18 is 2*10^6.
            break

    print("-1")

solve()