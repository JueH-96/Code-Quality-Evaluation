import sys
input = sys.stdin.read

MOD = 998244353

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    X = []
    Y = []
    idx = 3
    for _ in range(M):
        x = int(data[idx]) - 1
        y = int(data[idx+1]) - 1
        X.append(x % N)
        Y.append(y % N)
        idx += 2
    
    # Initialize f array
    f = [0] * N
    f[0] = 1  # transformed initial position is 0-based 0
    
    # Initialize pos and c arrays for each optional edge
    pos = [0] * M
    c = [0] * M
    for i in range(M):
        pos[i] = (X[i] + 1) % N
        c[i] = Y[i] % N
    
    for _ in range(K):
        updates = dict()
        # Collect contributions
        for i in range(M):
            val = f[pos[i]]
            current_c = c[i]
            if current_c in updates:
                updates[current_c] = (updates[current_c] + val) % MOD
            else:
                updates[current_c] = val % MOD
        
        # Apply updates
        for key in updates:
            f[key] = (f[key] + updates[key]) % MOD
        
        # Update pos and c for next step
        for i in range(M):
            pos[i] = (pos[i] - 1) % N
            c[i] = (c[i] - 1) % N
    
    # Sum all elements in f
    print(sum(f) % MOD)

if __name__ == "__main__":
    main()