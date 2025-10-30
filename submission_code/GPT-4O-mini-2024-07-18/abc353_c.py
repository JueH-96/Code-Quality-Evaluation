def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total_sum = 0
    MOD = 10**8
    
    for i in range(N):
        for j in range(i + 1, N):
            total_sum += (A[i] + A[j]) % MOD
            
    print(total_sum)

if __name__ == "__main__":
    main()