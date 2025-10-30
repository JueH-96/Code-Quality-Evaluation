def main():
    MOD = 10**8
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    
    result = 0
    for i in range(N-1):
        for j in range(i+1, N):
            result += (A[i] + A[j]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()