import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1
    extra_edges = []
    for _ in range(M):
        X = int(input[ptr]); ptr += 1
        Y = int(input[ptr]); ptr += 1
        extra_edges.append((X, Y))
    
    delta_prev = defaultdict(int)
    
    for step in range(1, K + 1):
        pos_prev = ((step - 1) % N) + 1
        
        # Rotate delta_prev
        rotated = defaultdict(int)
        for u in delta_prev:
            val = delta_prev[u]
            if u < N:
                new_u = u + 1
            else:
                new_u = 1
            rotated[new_u] = (rotated[new_u] + val) % MOD
        
        # Compute extra_increments
        extra_increments = defaultdict(int)
        for (u, v) in extra_edges:
            base_contribution = 1 if u == pos_prev else 0
            delta_contribution = delta_prev.get(u, 0)
            total = (base_contribution + delta_contribution) % MOD
            extra_increments[v] = (extra_increments[v] + total) % MOD
        
        # Combine rotated and extra_increments
        new_delta = rotated
        for v in extra_increments:
            new_delta[v] = (new_delta[v] + extra_increments[v]) % MOD
        
        # Update delta_prev
        delta_prev = new_delta
    
    # Calculate the answer
    total = (1 + sum(delta_prev.values())) % MOD
    print(total)

if __name__ == '__main__':
    main()