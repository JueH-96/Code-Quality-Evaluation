mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    total = 0
    for i in range(61):
        if (M >> i) & 1:
            base = 1 << i
            base2 = base << 1
            full_cycles = (N + 1) // base2
            remainder = (N + 1) % base2
            term1 = full_cycles * base
            term2 = remainder - base if remainder > base else 0
            total = (total + term1 + term2) % mod
    print(total)

if __name__ == '__main__':
    main()