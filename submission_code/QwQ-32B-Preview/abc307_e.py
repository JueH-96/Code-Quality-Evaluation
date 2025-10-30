def main():
    MOD = 998244353
    input_line = input().strip()
    N_M = input_line.split()
    N = int(N_M[0])
    M = int(N_M[1])
    
    if N % 2 == 0:
        answer = (pow(M - 1, N, MOD) + (M - 1)) % MOD
    else:
        answer = (pow(M - 1, N, MOD) - (M - 1) + MOD) % MOD
    print(answer)

if __name__ == "__main__":
    main()