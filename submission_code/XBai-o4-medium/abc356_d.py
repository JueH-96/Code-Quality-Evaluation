MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.read().split())
    total = 0
    for i in range(61):  # since bits are 0 to 60
        if (M >> i) & 1:
            cycle_len = 1 << (i + 1)
            full = N // cycle_len
            cnt = full * (1 << i)
            rem = N % cycle_len
            add = rem - (1 << i) + 1
            if add < 0:
                add = 0
            cnt += add
            total += cnt
    print(total % MOD)

if __name__ == "__main__":
    main()