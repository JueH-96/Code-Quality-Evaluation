# YOUR CODE HERE
MOD = 10**8

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    
    total_sum = 0
    prefix_sum = [0] * (N + 1)
    
    for i in range(N):
        prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % MOD
    
    for i in range(N):
        total_sum += (A[i] * (N - i - 1) + prefix_sum[N] - prefix_sum[i + 1]) % MOD
        total_sum %= MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()