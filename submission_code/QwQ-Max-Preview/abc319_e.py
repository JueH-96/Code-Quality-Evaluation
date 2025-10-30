import sys
import math

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    X = int(input[ptr]); ptr += 1
    Y = int(input[ptr]); ptr += 1

    P = []
    T = []
    for _ in range(N-1):
        p = int(input[ptr]); ptr += 1
        t = int(input[ptr]); ptr += 1
        P.append(p)
        T.append(t)
    
    # Compute LCM of all P_i
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    current_lcm = 1
    for p in P:
        current_lcm = lcm(current_lcm, p)
    L = current_lcm

    sum_T = sum(T)

    # Precompute sum_waiting for each residue
    L_size = L
    sum_waiting = [0] * L_size
    for r in range(L_size):
        current_r = r
        total = 0
        for i in range(len(P)):
            p = P[i]
            t = T[i]
            wt = (-current_r) % p
            total += wt
            current_r = (current_r + wt + t) % L_size
        sum_waiting[r] = total

    # Process queries
    Q = int(input[ptr]); ptr += 1
    output = []
    for _ in range(Q):
        q = int(input[ptr]); ptr += 1
        t_initial = q + X
        r = t_initial % L_size
        res = t_initial + sum_T + sum_waiting[r] + Y
        output.append(str(res))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()