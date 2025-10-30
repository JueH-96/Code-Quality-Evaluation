# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    MOD = 998244353
    
    N = int(input().strip())
    
    # Precompute factorials and powers of 10
    factorial = 1
    for i in range(2, N):
        factorial = (factorial * i) % MOD
    
    # Precompute powers of 10
    power_of_10 = [1] * N
    for i in range(1, N):
        power_of_10[i] = (power_of_10[i-1] * 10) % MOD
    
    # Calculate the sum
    total_sum = 0
    for digit in range(1, N + 1):
        for pos in range(N):
            contribution = (digit * power_of_10[pos] * factorial) % MOD
            total_sum = (total_sum + contribution) % MOD
    
    # Output the result
    print(total_sum)

main()