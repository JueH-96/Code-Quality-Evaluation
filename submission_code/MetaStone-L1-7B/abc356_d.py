def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    mod = 998244353
    total = 0
    for i in range(61):  # Since M can be up to 2^60-1, bits 0 to 60 are possible
        bit = 1 << i
        if (M & bit) == 0:
            continue
        pow2i = bit
        pow2i_plus1 = pow2i << 1
        full_blocks = (N + 1) // pow2i_plus1
        remainder = (N + 1) % pow2i_plus1
        count = full_blocks * pow2i
        if remainder > pow2i:
            count += remainder - pow2i
        total += count
        if total >= mod:
            total %= mod
    print(total % mod)

if __name__ == '__main__':
    main()