def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    from collections import defaultdict

    # Precompute f(x) for all x up to 2 * 10^7
    max_x = 2 * 10**7
    f = [0] * (max_x + 1)
    for x in range(1, max_x + 1):
        f[x] = x >> (x & -x).bit_length() - 1

    # Frequency map for A
    count = defaultdict(int)
    for a in A:
        count[a] += 1

    total = 0
    # Iterate over m from 0 to 30
    for m in range(31):
        M = 1 << (m + 1)
        freq_m = defaultdict(int)
        for a in A:
            freq_m[a % M] += 1
        for a in freq_m:
            b = (1 << m) - a
            if b < 0:
                b += M
            b %= M
            if a <= b:
                if a == b:
                    # Pairs where i == j
                    c = freq_m[a]
                    pairs = c * (c - 1) // 2
                    total += pairs * f[(a + b) // (1 << m)]
                else:
                    # Pairs where i < j
                    c1 = freq_m[a]
                    c2 = freq_m[b]
                    pairs = c1 * c2
                    total += pairs * f[(a + b) // (1 << m)]
    print(total)

if __name__ == '__main__':
    main()