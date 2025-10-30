import math

def multinomial(n, groups):
    """Calculate multinomial coefficient n! / (k1! * k2! * ... * km!)"""
    result = math.factorial(n)
    for k in groups:
        result //= math.factorial(k)
    return result

def solve(N, K):
    # Total number of good sequences
    total_sequences = multinomial(N * K, [K] * N)
    
    # Target position (1-indexed)
    target = (total_sequences + 1) // 2
    
    # Current state: how many times each number (1 to N) has been used
    remaining = [K] * N
    sequence = []
    
    # Total remaining positions to fill
    remaining_positions = N * K
    
    for pos in range(N * K):
        # For each possible number at this position
        for num in range(1, N + 1):
            if remaining[num - 1] == 0:
                continue
                
            # Count sequences that start with this number at current position
            remaining[num - 1] -= 1
            remaining_positions -= 1
            
            if remaining_positions == 0:
                count = 1
            else:
                # Calculate how many ways to arrange remaining numbers
                count = multinomial(remaining_positions, remaining)
            
            if target <= count:
                # This is the number we want at this position
                sequence.append(num)
                break
            else:
                # Skip this many sequences and try next number
                target -= count
                remaining[num - 1] += 1
                remaining_positions += 1
    
    return sequence

# Read input
N, K = map(int, input().split())

# Solve and output
result = solve(N, K)
print(' '.join(map(str, result)))