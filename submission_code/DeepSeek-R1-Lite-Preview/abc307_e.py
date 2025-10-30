def main():
    MOD = 998244353
    N, M = map(int, input().split())
    
    power = pow(M - 1, N, MOD)
    
    if N % 2 == 0:
        ans = (power + (M - 1)) % MOD
    else:
        ans = (power - (M - 1) + MOD) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()