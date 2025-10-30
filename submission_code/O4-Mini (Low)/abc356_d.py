def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    mod = 998244353

    ans = 0
    # For each bit i where M has a 1, add the count of k in [0..N] with bit i == 1
    for i in range(60):  # up to bit 59 is enough for 2^60 range
        if (M >> i) & 1:
            block = 1 << (i + 1)      # cycle length
            half = 1 << i             # number of ones per full cycle
            full_cycles = (N + 1) // block
            rem = (N + 1) % block
            ones = full_cycles * half + max(0, rem - half)
            ans = (ans + ones) % mod

    print(ans)

if __name__ == "__main__":
    main()