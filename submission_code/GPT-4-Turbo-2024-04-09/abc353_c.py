def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    MOD = 10**8
    result = 0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            result += (A[i] + A[j]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()