MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    edges = []
    for _ in range(M):
        X = int(input[ptr])
        ptr += 1
        Y = int(input[ptr])
        ptr += 1
        edges.append((X, Y))
    
    A = [0] * N
    A[0] = 1  # vertex 1 is index 0
    O = 0  # current offset
    sum_S = 1  # initial sum is 1
    
    for _ in range(K):
        O_prev = O
        O = (O + 1) % N
        
        # Compute sum_extra
        sum_extra = 0
        for X, Y in edges:
            X_real_prev = X - 1
            index_prev = (X_real_prev - O_prev) % N
            sum_extra = (sum_extra + A[index_prev]) % MOD
        sum_S = (sum_S + sum_extra) % MOD
        
        # Update A for each edge
        for X, Y in edges:
            X_real_prev = X - 1
            Y_real_current = Y - 1
            index_prev = (X_real_prev - O_prev) % N
            val = A[index_prev]
            index_current = (Y_real_current - O) % N
            A[index_current] = (A[index_current] + val) % MOD
    
    print(sum_S % MOD)

if __name__ == '__main__':
    main()