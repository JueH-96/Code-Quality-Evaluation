def solve():
    import sys
    from fractions import Fraction
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    persons = []
    idx = 1
    for i in range(1, N+1):
        A_i = int(input_data[idx]); B_i = int(input_data[idx+1])
        idx += 2
        # Store (person_number, heads, tails)
        persons.append((i, A_i, B_i))
    
    # Sort by descending success rate = A_i/(A_i+B_i),
    # tie-break by person number ascending
    # Use Fraction to avoid floating inaccuracies:
    persons.sort(key=lambda x: (-Fraction(x[1], x[1] + x[2]), x[0]))
    
    # Output the sorted person numbers
    print(" ".join(str(p[0]) for p in persons))

# Let's call solve() to process input and produce output
# (Comment out if running on an online judge that calls solve() automatically)
# solve()