# YOUR CODE HERE
def solve(L, R):
    result = []
    current = L
    
    while current < R:
        # Find the largest power of 2 such that:
        # 1. current is divisible by 2^i
        # 2. current + 2^i <= R
        
        max_i = 0
        
        # Find the largest i such that current is divisible by 2^i
        temp = current
        if temp == 0:
            # Special case: 0 is divisible by any power of 2
            # Find largest i such that 2^i <= R - current
            remaining = R - current
            i = 0
            while (1 << (i + 1)) <= remaining:
                i += 1
            max_i = i
        else:
            # Find largest power of 2 that divides current
            while temp % 2 == 0:
                temp //= 2
                max_i += 1
            
            # Now check if we can use this power of 2
            # We need current + 2^max_i <= R
            while max_i >= 0 and current + (1 << max_i) > R:
                max_i -= 1
        
        # Add the good sequence [current, current + 2^max_i)
        next_pos = current + (1 << max_i)
        result.append((current, next_pos))
        current = next_pos
    
    return result

# Read input
L, R = map(int, input().split())

# Solve
sequences = solve(L, R)

# Output
print(len(sequences))
for l, r in sequences:
    print(l, r)