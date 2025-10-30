import sys
from math import gcd

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    Y = int(input[ptr])
    ptr += 1
    
    buses = []
    M = 1  # Initialize LCM
    for _ in range(N-1):
        P = int(input[ptr])
        ptr += 1
        T = int(input[ptr])
        ptr += 1
        buses.append((P, T))
        # Compute LCM of M and P
        M = M * P // gcd(M, P) if M != 0 else P
    
    # Precompute F for t in 0 to M-1
    M = max(M, 1)  # Ensure M is at least 1
    F = [0] * M
    for t in range(M):
        current = t
        for (P, T) in buses:
            rem = current % P
            wait = (P - rem) % P
            depart = current + wait
            current = depart + T
        F[t] = current
    
    Q = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(Q):
        q = int(input[ptr])
        ptr += 1
        t0 = q + X
        rem = t0 % M
        cycles = t0 // M
        bus_time = F[rem] + cycles * M
        results.append(bus_time + Y)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()