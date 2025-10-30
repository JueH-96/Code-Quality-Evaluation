MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    try:
        N = int(data[0])
        M = int(data[1])
    except Exception:
        print(0)
        return
        
    ans = 0
    for i in range(61):
        if (M >> i) & 1:
            period = 1 << (i + 1)
            half = period // 2
            total = N + 1
            full_cycles = total // period
            remainder = total % period
            cnt = full_cycles * half
            if remainder > half:
                cnt += remainder - half
            ans = (ans + cnt) % MOD
            
    print(ans)

if __name__ == "__main__":
    main()