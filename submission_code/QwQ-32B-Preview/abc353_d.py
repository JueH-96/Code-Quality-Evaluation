def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Compute the number of digits for each A[j]
    d = [len(str(A[j])) for j in range(N)]
    
    # Compute suffix sums S[i] = σ_{j=i+1}^{N} 10^{d_j}
    S = [0] * N
    sum_suffix = 0
    for j in range(N-1, 0, -1):
        sum_suffix += 10 ** d[j]
        S[j-1] = sum_suffix
    
    # Compute the first sum: σ_{i=1}^{N-1} A_i * S_i
    sum1 = 0
    for i in range(N-1):
        sum1 += A[i] * S[i]
        sum1 %= 998244353
    
    # Compute the second sum: σ_{j=2}^{N} A_j * (j - 1)
    sum2 = 0
    for j in range(1, N):
        sum2 += A[j] * (j)
        sum2 %= 998244353
    
    # Total sum
    total_sum = (sum1 + sum2) % 998244353
    print(total_sum)

if __name__ == '__main__':
    main()