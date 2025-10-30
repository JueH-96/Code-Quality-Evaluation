MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    s = input[1] if len(input) > 1 else ""
    k = s.count('1')
    pow2k = pow(2, k, MOD)
    pow2N = pow(2, N, MOD)
    ans = (pow2k * (pow2N - 1)) % MOD
    print(ans)

if __name__ == "__main__":
    main()