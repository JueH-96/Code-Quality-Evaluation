def solve(L, R):
    result = []
    current = L
    
    while current < R:
        # Find the largest power of 2 that we can use
        max_power = 0
        
        # Check powers of 2 from large to small
        for i in range(60, -1, -1):
            power_of_2 = 1 << i  # 2^i
            
            # Check if we can use this power of 2
            # 1. The sequence shouldn't exceed R
            if current + power_of_2 > R:
                continue
                
            # 2. current must be divisible by power_of_2
            if current % power_of_2 == 0:
                max_power = power_of_2
                break
        
        # If current is not divisible by any power of 2 that fits,
        # we need to find the largest power of 2 that divides current
        if max_power == 0:
            # Find the largest power of 2 that divides current
            if current == 0:
                # Special case: 0 is divisible by any power of 2
                for i in range(60, -1, -1):
                    power_of_2 = 1 << i
                    if current + power_of_2 <= R:
                        max_power = power_of_2
                        break
            else:
                # Find the largest power of 2 that divides current
                temp = current
                max_power = 1
                while temp % 2 == 0 and current + max_power * 2 <= R:
                    temp //= 2
                    max_power *= 2
        
        # Add the sequence to result
        result.append((current, current + max_power))
        current += max_power
    
    return result

# Read input
L, R = map(int, input().split())

# Solve the problem
sequences = solve(L, R)

# Print output
print(len(sequences))
for l, r in sequences:
    print(l, r)