MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    prefix_sum = [0] * (N + 1)
    for i in range(1, N+1):
        prefix_sum[i] = (prefix_sum[i-1] + A[i-1]) % MOD
    
    first_part = 0
    for j in range(N):
        a = A[j]
        length = len(str(a))
        pow10 = pow(10, length, MOD)
        term = (prefix_sum[j] * pow10) % MOD
        first_part = (first_part + term) % MOD
    
    second_part = 0
    for j in range(N):
        term = (j * A[j]) % MOD
        second_part = (second_part + term) % MOD
    
    total = (first_part + second_part) % MOD
    print(total)

if __name__ == "__main__":
    main()