def is_perfect_power(n):
    """Check if n can be expressed as a^b for some a > 1 and b > 1"""
    if n <= 1:
        return False
    
    # Check for each possible exponent b
    for b in range(2, 64):  # log2(10^18) < 60
        # Binary search for a such that a^b = n
        left, right = 1, int(n**(1/b)) + 2
        while left <= right:
            mid = (left + right) // 2
            val = mid ** b
            if val == n:
                return True
            elif val < n:
                left = mid + 1
            else:
                right = mid - 1
        
        # If a^b would be less than 2^b, no point checking further
        if 2**b > n:
            break
    
    return False

def count_perfect_powers(N):
    # Use a set to store all perfect powers to avoid duplicates
    perfect_powers = set()
    
    # For each exponent b >= 2
    b = 2
    while True:
        # Find maximum a such that a^b <= N
        max_a = int(N**(1/b))
        
        # Verify and adjust max_a if needed due to floating point errors
        while max_a > 0 and max_a**b > N:
            max_a -= 1
        while (max_a + 1)**b <= N:
            max_a += 1
        
        if max_a < 2:
            break
        
        # Add all a^b where a is not already a perfect power
        for a in range(2, max_a + 1):
            val = a ** b
            if val <= N:
                perfect_powers.add(val)
        
        b += 1
    
    # 1 is a special case: 1 = 1^b for any b >= 2
    if N >= 1:
        perfect_powers.add(1)
    
    return len(perfect_powers)

# Read input
N = int(input())

# Calculate and print the answer
print(count_perfect_powers(N))