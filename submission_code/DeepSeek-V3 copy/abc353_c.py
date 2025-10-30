def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    MOD = 10**8
    total = 0
    
    # Precompute the sum of all A_i
    sum_A = sum(A)
    
    # Compute the sum of all pairs (A_i + A_j) mod MOD
    # The sum of all pairs (A_i + A_j) is sum_A * (N-1) - sum_{i=1}^N A_i * (N - i)
    # But to compute the sum of (A_i + A_j) mod MOD, we need to handle it differently
    
    # Instead, we can compute the sum of all pairs (A_i + A_j) and then take mod MOD
    # But since N is up to 3e5, we need an efficient way
    
    # The sum of all pairs (A_i + A_j) is sum_A * (N-1) - sum_{i=1}^N A_i * (N - i)
    # But to compute the sum of (A_i + A_j) mod MOD, we can compute the sum of all pairs and then take mod MOD
    
    # Compute the sum of all pairs (A_i + A_j)
    sum_pairs = sum_A * (N-1)
    
    # Now, compute the sum of (A_i + A_j) mod MOD
    # Since (A_i + A_j) mod MOD = (A_i mod MOD + A_j mod MOD) mod MOD
    # We can precompute A_i mod MOD for all i
    A_mod = [a % MOD for a in A]
    
    # Now, compute the sum of (A_i + A_j) mod MOD
    # We can use the fact that (a + b) mod MOD = (a mod MOD + b mod MOD) mod MOD
    # So, sum_{i<j} (A_i + A_j) mod MOD = sum_{i<j} (A_i mod MOD + A_j mod MOD) mod MOD
    
    # To compute this efficiently, we can count the number of pairs where (A_i mod MOD + A_j mod MOD) >= MOD
    # and subtract MOD * count from the total sum
    
    # First, compute the sum of all (A_i mod MOD + A_j mod MOD)
    sum_mod_pairs = sum(A_mod) * (N-1)
    
    # Now, count the number of pairs where (A_i mod MOD + A_j mod MOD) >= MOD
    # To do this, we can sort A_mod and use a two-pointer approach
    
    A_mod_sorted = sorted(A_mod)
    count = 0
    left = 0
    right = N - 1
    
    while left < right:
        if A_mod_sorted[left] + A_mod_sorted[right] >= MOD:
            count += right - left
            right -= 1
        else:
            left += 1
    
    # The total sum is sum_mod_pairs - MOD * count
    total = sum_mod_pairs - MOD * count
    
    print(total)

if __name__ == "__main__":
    main()