import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    used = set()
    conflict = False

    # Check for conflicts in A and B
    for a in A:
        if a in used:
            conflict = True
        used.add(a)
    
    fixed_B = []
    for b in B:
        if b != -1:
            if b in used:
                conflict = True
            used.add(b)
            fixed_B.append(b)
        else:
            fixed_B.append(-1)
    
    # Check if any A[i] == B[i] where B[i] is not -1
    for i in range(N):
        if B[i] != -1 and A[i] == B[i]:
            conflict = True
    
    if conflict:
        print(0)
        return
    
    # Collect all used values
    used_values = set(A)
    for b in B:
        if b != -1:
            used_values.add(b)
    
    max_val = 2 * N
    F = [v for v in range(1, max_val + 1) if v not in used_values]
    M = len(F)
    K = B.count(-1)
    
    # Collect forbidden values for each free variable
    forbidden_list = []
    free_indices = []
    for i in range(N):
        if B[i] == -1:
            forbidden = A[i]
            forbidden_in_F = forbidden in F
            forbidden_list.append(forbidden)
            free_indices.append(i)
        else:
            forbidden_list.append(None)
    
    C = 0
    forbidden_set = set()
    for i in range(N):
        if B[i] == -1:
            f = A[i]
            if f in used_values:
                continue
            if f in F:
                if f in forbidden_set:
                    # Forbidden value appears more than once, which invalidates the assumption of unique forbidden values
                    print(0)
                    return
                forbidden_set.add(f)
                C += 1
    
    # Precompute factorials and inverse factorials
    max_fact = 2 * N + 10
    fact = [1] * (max_fact)
    for i in range(1, max_fact):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_fact)
    inv_fact[max_fact-1] = pow(fact[max_fact-1], MOD-2, MOD)
    for i in range(max_fact-2, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    def perm(n, r):
        if r < 0 or r > n:
            return 0
        return fact[n] * inv_fact[n - r] % MOD
    
    # Inclusion-exclusion sum
    res = 0
    for k in range(0, C + 1):
        comb_c_k = fact[C] * inv_fact[k] % MOD * inv_fact[C - k] % MOD
        term = comb_c_k * perm(M - k, K - k) % MOD
        if k % 2 == 1:
            term = (-term + MOD) % MOD
        res = (res + term) % MOD
    
    print(res)

threading.Thread(target=main).start()