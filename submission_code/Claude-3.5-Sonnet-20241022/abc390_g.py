def solve():
    MOD = 998244353
    
    # Read input
    N = int(input())
    
    # For N ≤ 10, we can calculate directly using permutations
    if N <= 10:
        from itertools import permutations
        
        def f(perm):
            s = ''.join(map(str, perm))
            return int(s)
        
        result = sum(f(p) for p in permutations(range(1, N+1)))
        print(result % MOD)
        return
    
    # For larger N, we need a mathematical approach
    # For each position, calculate how many times each digit appears there
    # and its contribution to the final sum
    
    # First, calculate number of digits for each number from 1 to N
    digit_count = [len(str(i)) for i in range(N+1)]
    
    # Calculate factorial modulo MOD
    fact = [1]
    for i in range(1, N+1):
        fact.append((fact[-1] * i) % MOD)
    
    # Calculate inverse modulo using Fermat's little theorem
    def mod_inverse(x):
        return pow(x, MOD-2, MOD)
    
    # For each number from 1 to N, calculate its contribution
    result = 0
    
    # For each number i from 1 to N
    for i in range(1, N+1):
        num_str = str(i)
        num_len = digit_count[i]
        
        # For each possible position where this number can start
        for pos in range(N):
            # Calculate how many numbers will be before and after this number
            numbers_before = pos
            numbers_after = N - pos - 1
            
            # Calculate multiplier for this position
            # It will be (N-1)! divided by position arrangements
            multiplier = fact[N-1]
            
            # Calculate power of 10 for this position
            # Sum of lengths of all numbers that will come after this position
            total_after_len = 0
            for j in range(1, N+1):
                if j != i:
                    total_after_len += digit_count[j]
            
            power = total_after_len * numbers_before * mod_inverse(N-1)
            power %= MOD
            
            # Add contribution
            contribution = (int(num_str) * pow(10, power, MOD)) % MOD
            contribution = (contribution * multiplier) % MOD
            
            result = (result + contribution) % MOD
    
    print(result)

solve()