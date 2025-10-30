def solve(N, M):
    MOD = 998244353
    result = 0
    
    for i in range(60):  # Considering up to 60 bits (constraint says N, M can be up to 2^60-1)
        if M & (1 << i):  # Check if i-th bit of M is 1
            # Calculate how many numbers from 0 to N have a 1 in the i-th position
            # This is only relevant when M has a 1 in this position (due to bitwise AND)
            
            full_periods = N // (1 << (i + 1))
            remainder = N % (1 << (i + 1))
            
            # For each full period of 2^(i+1) numbers, exactly 2^i numbers have a 1 in the i-th position
            count = full_periods * (1 << i)
            
            # For the remaining numbers, we need to check if we've reached the part of the period
            # where the i-th bit is set
            if remainder >= (1 << i):
                count += remainder - (1 << i) + 1
            
            result = (result + count) % MOD
    
    return result

# Read input
N, M = map(int, input().split())
print(solve(N, M))