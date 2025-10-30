import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.read().split())
    L = N * K
    # Precompute factorial up to L
    factorial = [1] * (L + 1)
    for i in range(1, L + 1):
        factorial[i] = factorial[i - 1] * i
    # Compute S = (N*K)! / (K!)^N
    fact_K = factorial[K]
    denominator = 1
    for _ in range(N):
        denominator *= fact_K
    S = factorial[L] // denominator
    M = (S + 1) // 2
    # Initialize counts
    counts = {i: K for i in range(1, N + 1)}
    sequence = []
    # Initialize C = S
    C = S
    remaining_length = L
    while remaining_length > 0:
        for num in range(1, N + 1):
            if counts[num] == 0:
                continue
            # Compute C_i = C * counts[num] // remaining_length
            C_i = C * counts[num] // remaining_length
            if M > C_i:
                M -= C_i
            else:
                sequence.append(num)
                counts[num] -= 1
                C = C_i
                remaining_length -= 1
                break
    print(' '.join(map(str, sequence)))

if __name__ == '__main__':
    main()