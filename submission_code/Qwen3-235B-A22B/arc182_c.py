import sys

MOD = 998244353

def mat_mult(a, b):
    n = len(a)
    m = len(b[0])
    p = len(b)
    res = [[0]*m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            if a[i][k]:
                for j in range(m):
                    res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % MOD
    return res

def mat_pow(mat, power):
    n = len(mat)
    res = [[int(i==j) for j in range(n)] for i in range(n)]
    while power > 0:
        if power % 2 == 1:
            res = mat_mult(mat, res)
        mat = mat_mult(mat, mat)
        power //= 2
    return res

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    # Generate primes up to M
    primes = []
    is_prime = [False] * (M+1)
    if M >= 2:
        is_prime[2] = True
    for i in range(3, M+1, 2):
        is_prime[i] = True
    for i in range(3, int(M**0.5)+1, 2):
        if is_prime[i]:
            for j in range(i*i, M+1, i):
                is_prime[j] = False
    primes = []
    for i in range(2, M+1):
        if is_prime[i]:
            primes.append(i)
    q = len(primes)
    if q == 0:
        print(0)
        return
    
    # Precompute exponents for each number and each prime
    e = []
    for x in range(M+1):
        e.append([0]*q)
        for i, p in enumerate(primes):
            cnt = 0
            val = x
            while val % p == 0 and val > 0:
                cnt += 1
                val //= p
            e[x][i] = cnt
    
    total = 0
    
    # Iterate over all subsets of primes (including empty subset)
    from itertools import combinations
    
    for subset_mask in range(1 << q):
        S = []
        for i in range(q):
            if (subset_mask >> i) & 1:
                S.append(i)
        s = len(S)
        if s == 0:
            # Handle empty subset S separately
            if M == 1:
                sum_val = N % MOD
            else:
                # geometric series sum M*(M^N - 1)/(M-1)
                pow_m_n = pow(M, N, MOD)
                sum_val = (pow_m_n - 1) * M % MOD
                inv = pow(M - 1, MOD-2, MOD)
                sum_val = sum_val * inv % MOD
            total = (total + sum_val) % MOD
            continue
        
        # Generate all subsets of S (represented as bitmask of length s)
        num_states = 1 << s
        index_map = {}
        for i in range(num_states):
            index_map[i] = i
        
        # Precompute c[T] for all subsets T of S
        c = [0] * (1 << s)
        for t_mask in range(1 << s):
            T = []
            for i in range(s):
                if (t_mask >> i) & 1:
                    T.append(S[i])
            sum_c = 0
            for x in range(1, M+1):
                prod = 1
                for pi in T:
                    prod = (prod * e[x][pi]) % MOD
                sum_c = (sum_c + prod) % MOD
            c[t_mask] = sum_c
        
        # Build transition matrix trans: size (num_states) x (num_states)
        trans = [[0]*num_states for _ in range(num_states)]
        for u in range(num_states):
            for v in range(u+1):
                if (v | u) != u:
                    continue
                diff = u ^ v
                trans[v][u] = (trans[v][u] + c[diff]) % MOD
        
        # Build augmented matrix
        aug_size = num_states + 1
        aug = [[0]*aug_size for _ in range(aug_size)]
        # Copy trans into aug[0..num_states-1][0..num_states-1]
        for i in range(num_states):
            for j in range(num_states):
                aug[i][j] = trans[i][j]
        # Last row: sum += current S and carry previous sum
        s_mask = (1 << s) - 1
        s_idx = s_mask
        for j in range(num_states):
            aug[num_states][j] = 0
        aug[num_states][s_idx] = 1
        aug[num_states][num_states] = 1
        
        # Initial augmented vector: initial length 0, sum 0
        vec = [0] * aug_size
        # initial vector: subset_mask 0 (only the empty set)
        vec[0] = 1
        vec[-1] = 0
        
        # Compute matrix exponentiation
        powered = mat_pow(aug, N)
        
        # Multiply powered matrix with initial vector
        new_vec = [0] * aug_size
        for i in range(aug_size):
            if vec[i] == 0:
                continue
            for j in range(aug_size):
                new_vec[j] = (new_vec[j] + powered[j][i] * vec[i]) % MOD
        
        # Extract cumulative sum
        g_S = new_vec[-1]
        total = (total + g_S) % MOD
    
    print(total % MOD)

if __name__ == '__main__':
    main()