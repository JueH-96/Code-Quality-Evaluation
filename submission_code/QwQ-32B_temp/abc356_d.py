mod = 998244353

def main():
    N, M = map(int, input().split())
    total = 0
    for i in range(61):
        if M & (1 << i):
            high = 1 << i
            cycle = 1 << (i + 1)
            full_cycles = (N + 1) // cycle
            remainder = (N + 1) % cycle
            cnt = full_cycles * high + max(0, remainder - high)
            total += cnt
    print(total % mod)

if __name__ == "__main__":
    main()