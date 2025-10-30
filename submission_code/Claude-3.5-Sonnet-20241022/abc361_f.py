def solve():
    N = int(input())
    
    # Use set to avoid duplicates
    powers = set()
    
    # For each base a
    a = 2
    while a * a <= N:  # a^2 needs to be <= N
        # For each exponent b
        b = 2
        val = a ** b
        while val <= N:
            powers.add(val)
            b += 1
            val = a ** b
        a += 1
    
    # Handle case of a > sqrt(N)
    # These can only have exponent 2 since a^3 would exceed N
    a = int(N ** 0.5)
    while a * a <= N:
        powers.add(a * a)
        a += 1
    
    # Add 1 since 1 can be expressed as any number to any power
    powers.add(1)
    
    print(len(powers))

solve()