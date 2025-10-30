def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    MOD = 998244353
    total_sum = 0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Calculate f(A[i], A[j])
            f_ij = int(str(A[i]) + str(A[j]))
            total_sum = (total_sum + f_ij) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()