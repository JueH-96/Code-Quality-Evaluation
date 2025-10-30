import sys
from itertools import combinations

def main():
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # We know C(N, K) <= 1e6, so brute‐force all combinations works in time.
    best = 0

    # It's slightly faster to iterate the smaller of K and N-K,
    # so if K > N-K we iterate over complements of size N-K.
    if K <= N - K:
        # enumerate all K‐subsets
        for comb in combinations(A, K):
            x = 0
            for v in comb:
                x ^= v
            if x > best:
                best = x
    else:
        # enumerate all (N-K)‐subsets as the complement,
        # then the XOR of the chosen K elements is total_xor ^ xor_of_complement
        total_xor = 0
        for v in A:
            total_xor ^= v
        R = N - K
        for comb in combinations(A, R):
            y = 0
            for v in comb:
                y ^= v
            x = total_xor ^ y
            if x > best:
                best = x

    print(best)

if __name__ == "__main__":
    main()