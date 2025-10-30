MOD = 10**9 + 7
P = 1000003

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    
    # Precompute P^k % MOD for k from 1 to N
    P_pow = [1] * (N + 1)
    for k in range(1, N + 1):
        P_pow[k] = (P_pow[k - 1] * P) % MOD
    
    # Compute prefix hash for A
    hash_A = [0] * (N + 1)
    for i in range(1, N + 1):
        hash_A[i] = (hash_A[i - 1] + P_pow[A[i - 1]]) % MOD
    
    # Compute prefix hash for B
    hash_B = [0] * (N + 1)
    for i in range(1, N + 1):
        hash_B[i] = (hash_B[i - 1] + P_pow[B[i - 1]]) % MOD
    
    # Process Q queries
    index = 2 + 2 * N
    for _ in range(Q):
        l = int(data[index]) - 1  # Convert to 0-based indexing
        r = int(data[index + 1]) - 1
        L = int(data[index + 2]) - 1
        R = int(data[index + 3]) - 1
        index += 4
        
        # Compute hash for A[l to r]
        hash_sub_A = (hash_A[r + 1] - hash_A[l] + MOD) % MOD
        
        # Compute hash for B[L to R]
        hash_sub_B = (hash_B[R + 1] - hash_B[L] + MOD) % MOD
        
        if hash_sub_A == hash_sub_B:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()