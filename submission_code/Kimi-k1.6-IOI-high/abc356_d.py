MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    ans = 0
    for i in range(60):
        if (M >> i) & 1:
            cycle = 1 << (i + 1)
            full_cycles = (N + 1) // cycle
            cnt = full_cycles * (1 << i)
            remainder = (N + 1) % cycle
            if remainder > (1 << i):
                cnt += remainder - (1 << i)
            ans += cnt
    print(ans % MOD)

if __name__ == "__main__":
    main()