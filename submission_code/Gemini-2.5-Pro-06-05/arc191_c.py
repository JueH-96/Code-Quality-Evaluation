import sys

def main():
    """
    Reads input, solves for each test case, and prints the output.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str:
            return
        T = int(T_str)
    except (IOError, ValueError):
        return

    for _ in range(T):
        try:
            N_str = sys.stdin.readline()
            if not N_str:
                break
            N = int(N_str)
        except (IOError, ValueError):
            break

        # Handle the special case N=1
        if N == 1:
            print(2, 1)
            continue

        # 1. Prime factorization of N
        factors = {}
        temp_N = N
        d = 2
        while d * d <= temp_N:
            if temp_N % d == 0:
                count = 0
                while temp_N % d == 0:
                    count += 1
                    temp_N //= d
                factors[d] = count
            d += 1
        if temp_N > 1:
            factors[temp_N] = 1

        # 2. Construct (A_i, M_i) pairs for each prime power factor
        am_pairs = []
        if 2 in factors:
            e = factors[2]
            a = 5
            m = 1 << (e + 2)  # M_i = 2**(e+2)
            am_pairs.append((a, m))
            del factors[2]

        for p, e in factors.items():
            a = p + 1
            m = pow(p, e + 1) # M_i = p**(e+1)
            am_pairs.append((a, m))

        # 3. Solve the system of congruences using Chinese Remainder Theorem
        if not am_pairs:
            # This case is unreachable for N > 1
            continue
        
        current_a, current_m = am_pairs[0]

        for i in range(1, len(am_pairs)):
            next_a, next_m = am_pairs[i]
            
            # Solve for k in: current_a + k * current_m ≡ next_a (mod next_m)
            # k ≡ (next_a - current_a) * mod_inverse(current_m, next_m)
            inv = pow(current_m, -1, next_m)
            k = ((next_a - current_a) * inv) % next_m
            
            # Update the combined solution
            current_a = current_a + k * current_m
            current_m *= next_m
            
        A = current_a
        M = current_m
        
        print(A, M)

if __name__ == "__main__":
    main()