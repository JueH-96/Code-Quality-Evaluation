def sum_of_permutations(N):
    MOD = 998244353
    
    # Calculate the contribution of each digit position
    # The number of permutations is N!
    # Each number i (1 to N) will appear in each position (1 to N) equally across all permutations
    # Specifically, each number will appear (N-1)! times in each position
    
    # Precompute factorials and factorial inverses
    factorial = [1] * (N + 1)
    for i in range(2, N + 1):
        factorial[i] = factorial[i - 1] * i % MOD
    
    # (N-1)! is the number of times each number appears in each position
    count_per_position = factorial[N - 1]
    
    total_sum = 0
    power_of_ten = 1
    
    # Calculate the contribution of each position
    for position in range(N):
        # Each position contributes (1 + 2 + ... + N) * (N-1)! * 10^position
        # Sum of first N natural numbers is N * (N + 1) // 2
        sum_of_numbers = N * (N + 1) // 2
        contribution = sum_of_numbers * count_per_position % MOD
        contribution = contribution * power_of_ten % MOD
        
        total_sum = (total_sum + contribution) % MOD
        
        # Update power_of_ten for the next position
        power_of_ten = power_of_ten * 10 % MOD
    
    return total_sum

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Calculate and print the result
result = sum_of_permutations(N)
print(result)