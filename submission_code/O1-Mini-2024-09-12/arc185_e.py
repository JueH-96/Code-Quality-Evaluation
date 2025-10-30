# YOUR CODE HERE
import sys
import sys
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    max_A = max(A) if N >0 else 1
    # Precompute divisors for each number up to max_A
    divisors = [[] for _ in range(max_A +1)]
    for d in range(1, max_A +1):
        for multiple in range(d, max_A +1, d):
            divisors[multiple].append(d)
    # Sort divisors in descending order
    for d in range(1, max_A +1):
        divisors[d].sort(reverse=True)
    
    # Precompute powers of 2
    pow2 = [1]*(N+1)
    for i in range(1,N+1):
        pow2[i] = (pow2[i-1] *2) % MOD
    
    # Initialize F[d} =0
    F = [0]*(max_A +1)
    
    S_prev =0
    output = []
    for j in range(1, N+1):
        A_j = A[j-1]
        divs = divisors[A_j]
        C_j =0
        G = {}
        for d in divs:
            total = F[d]
            # Iterate over multiples m of d that are >d and divide A_j
            # Since divs are sorted descendingly, all m >d are already in G
            # So just iterate through divs and check if m is multiple of d and m >d
            for m in divs:
                if m > d and m % d ==0:
                    total = (total - G[m]) % MOD
            G[d] = total
            C_j = (C_j + d * G[d]) % MOD
        S_m = (2 * S_prev + C_j) % MOD
        output.append(str(S_m))
        # Update F[d} by adding 2^{j-1} for each d |A_j}
        add_val = pow2[j-1]
        for d in divs:
            F[d] = (F[d] + add_val) % MOD
        S_prev = S_m
    print('
'.join(output))

if __name__ == "__main__":
    main()