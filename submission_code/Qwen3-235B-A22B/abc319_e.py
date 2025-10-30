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
    sum_T = 0
    for _ in range(N - 1):
        p = int(input[ptr]); ptr += 1
        t = int(input[ptr]); ptr += 1
        P.append(p)
        T.append(t)
        sum_T += t

    # Compute LCM of all P
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    L = 1
    for p in P:
        g = gcd(L, p)
        L = (L // g) * p

    max_L = L
    wait_times = [0] * max_L

    # Precompute wait times for all residues mod L
    for r in range(max_L):
        current_mod = r
        total_wait = 0
        for i in range(N - 1):
            p = P[i]
            t = T[i]
            rem = current_mod % p
            if rem == 0:
                wait = 0
            else:
                wait = (p - rem) % p
            total_wait += wait
            departure_mod = (current_mod + wait) % L
            arrival_next_mod = (departure_mod + t) % L
            current_mod = arrival_next_mod
        wait_times[r] = total_wait

    Q = int(input[ptr]); ptr += 1
    output = []
    for _ in range(Q):
        q = int(input[ptr]); ptr += 1
        S = q + X
        r = S % L
        pre_wait = wait_times[r]
        arrival_stop_N = S + sum_T + pre_wait
        ans = arrival_stop_N + Y
        output.append(str(ans))

    print('
'.join(output))

if __name__ == "__main__":
    main()