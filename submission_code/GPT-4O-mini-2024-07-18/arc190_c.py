def main():
    import sys
    from sys import stdin, stdout
    from collections import defaultdict
    input = stdin.read
    data = input().splitlines()
    
    MOD = 998244353
    
    # Read dimensions
    H, W = map(int, data[0].split())
    
    # Read the grid
    A = [list(map(int, data[i + 1].split())) for i in range(H)]
    
    # Read starting position and number of queries
    Q, sh, sw = map(int, data[H + 1].split())
    sh -= 1  # Convert to 0-indexed
    sw -= 1  # Convert to 0-indexed
    
    # Read the queries
    queries = [data[H + 2 + i].split() for i in range(Q)]
    for i in range(Q):
        d_i, a_i = queries[i]
        a_i = int(a_i)
        
        # Move Takahashi according to the direction
        if d_i == 'L':
            sw -= 1
        elif d_i == 'R':
            sw += 1
        elif d_i == 'U':
            sh -= 1
        elif d_i == 'D':
            sh += 1
        
        # Update the grid
        A[sh][sw] = a_i
        
        # Calculate the sum of f(P) over all paths
        # We need to calculate the product of values along the paths
        # The number of paths from (1,1) to (H,W) is C(H+W-2, H-1)
        
        # Precompute factorials and inverse factorials for binomial coefficients
        max_factorial = H + W - 1
        fact = [1] * (max_factorial + 1)
        inv_fact = [1] * (max_factorial + 1)
        
        for j in range(2, max_factorial + 1):
            fact[j] = fact[j - 1] * j % MOD
        
        inv_fact[max_factorial] = pow(fact[max_factorial], MOD - 2, MOD)
        for j in range(max_factorial - 1, 0, -1):
            inv_fact[j] = inv_fact[j + 1] * (j + 1) % MOD
        
        # Calculate the number of paths
        num_paths = (fact[H + W - 2] * inv_fact[H - 1] % MOD) * inv_fact[W - 1] % MOD
        
        # Calculate the product of values along the paths
        product = 1
        for r in range(H):
            for c in range(W):
                if r + c < H + W - 1:
                    product = product * A[r][c] % MOD
        
        result = product * num_paths % MOD
        stdout.write(f"{result}
")

if __name__ == "__main__":
    main()