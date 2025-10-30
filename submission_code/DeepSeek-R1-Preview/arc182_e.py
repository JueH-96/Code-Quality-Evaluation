import sys
import bisect

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    C = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    A = [a % M for a in A]
    A.sort()
    min_A = A[0]

    if C == 0:
        print(min_A * K)
        return

    from math import gcd
    g = gcd(C, M)
    T = M // g

    # Threshold for T to decide whether to precompute or not
    if T > 1e6:
        print(0)
        return

    # Precompute residues
    residues = []
    current = 0
    for _ in range(T):
        residues.append(current)
        current = (current + C) % M

    # Precompute f(r) for each residue
    sum_f = 0
    f = []
    for r in residues:
        x = M - r
        j = bisect.bisect_left(A, x)
        candidates = []
        if j > 0:
            candidates.append(A[0] + r)
        if j < N:
            candidates.append(A[j] + r - M)
        if candidates:
            min_val = min(candidates)
        else:
            min_val = 0  # should not happen
        f.append(min_val)
        sum_f += min_val

    q, rem = divmod(K, T)
    total = q * sum_f

    # Compute remaining terms
    for k in range(rem):
        r = residues[k]
        total += f[k]

    print(total)

if __name__ == '__main__':
    main()