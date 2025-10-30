import sys
import math

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    Y = int(input[ptr])
    ptr += 1
    
    P = []
    T = []
    for _ in range(N-1):
        p = int(input[ptr])
        t = int(input[ptr+1])
        P.append(p)
        T.append(t)
        ptr += 2
    
    # Compute LCM of all P_i
    M = 1
    for p in P:
        M = M * p // math.gcd(M, p)
    
    # Precompute g[r] for each residue r modulo M
    g = [0] * M
    for r in range(M):
        current = r
        for i in range(len(P)):
            p = P[i]
            t_time = T[i]
            # Calculate departure time
            dep = (current + p - 1) // p * p
            current = dep + t_time
        g[r] = current - r
    
    # Process queries
    Q = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(Q):
        q_i = int(input[ptr])
        ptr += 1
        t = q_i + X
        r = t % M
        ans = t + g[r] + Y
        results.append(ans)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()