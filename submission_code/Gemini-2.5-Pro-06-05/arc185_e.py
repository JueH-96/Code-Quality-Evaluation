import sys

def get_divisors(n, spf):
    """
    Generates all divisors of n given the precomputed smallest prime factor sieve.
    This is an efficient method to get all divisors.
    """
    if n == 1:
        return [1]
    
    divs = [1]
    temp_n = n
    while temp_n > 1:
        p = spf[temp_n]
        # Store the current list of divisors before extending it
        # for the new prime factor p.
        len_divs = len(divs)
        
        # Power of p in the factorization of temp_n
        pk = 1
        while temp_n % p == 0:
            pk *= p
            # For each existing divisor, multiply by the current power of p
            # and add to the list.
            for i in range(len_divs):
                divs.append(divs[i] * pk)
            temp_n //= p
            
    return divs

def solve():
    """
    Main function to solve the problem.
    """
    # Fast I/O
    input = sys.stdin.readline
    
    try:
        N = int(input())
        if N == 0:
            return
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handles empty input at EOF for some online judges.
        return

    MOD = 998244353
    MAX_A = 100001
    
    # --- Precomputation ---
    
    # Sieve for phi (Euler's totient function)
    phi = list(range(MAX_A))
    for i in range(2, MAX_A):
        if phi[i] == i:  # i is a prime number
            for j in range(i, MAX_A, i):
                phi[j] -= phi[j] // i
    
    # Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_A))
    for i in range(2, int(MAX_A**0.5) + 1):
        if spf[i] == i:  # i is a prime
            for j in range(i * i, MAX_A, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Powers of 2 modulo MOD
    pow2 = [1] * N
    for i in range(1, N):
        pow2[i] = (pow2[i-1] * 2) % MOD
        
    # --- Main Logic ---
    
    s_current = 0  # This holds S_m at each step
    # U[d] stores sum_{j=0..m-1, d|A_j} 2^j
    U = [0] * MAX_A
    
    # Loop for m from 1 to N (using 0-indexed m from 0 to N-1)
    for m in range(N):
        v = A[m]
        
        # Get all divisors of v using the precomputed SPF sieve
        divs = get_divisors(v, spf)
        
        # Calculate T_m = sum_{d|v} phi[d] * U[d]
        t_m = 0
        for d in divs:
            term = (phi[d] * U[d]) % MOD
            t_m = (t_m + term) % MOD
        
        # Calculate S_m = 2 * S_{m-1} + T_m
        s_current = (2 * s_current + t_m) % MOD
        
        # Print the answer for the current prefix length m+1
        sys.stdout.write(str(s_current) + '
')
        
        # Update U for the next step.
        # For all d|v, U[d] += 2^m
        val_to_add = pow2[m]
        for d in divs:
            U[d] = (U[d] + val_to_add) % MOD

# Running the solver
if __name__ == "__main__":
    solve()