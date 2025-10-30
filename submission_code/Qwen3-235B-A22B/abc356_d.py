import sys

def main():
    MOD = 998244353
    N, M = map(int, sys.stdin.readline().split())
    result = 0
    for i in range(60):
        if (M >> i) & 1:
            mask = 1 << i
            cycle = mask << 1
            full_cycles = (N + 1) // cycle
            count = full_cycles * mask
            remainder = (N + 1) % cycle
            add = max(0, remainder - mask)
            count += add
            result += count
            result %= MOD
    print(result % MOD)

if __name__ == "__main__":
    main()