def solve():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, D = map(int, data)
    
    # Generate and collect the arithmetic sequence terms
    terms = [str(x) for x in range(A, B + 1, D)]
    
    # Print terms separated by spaces
    print(" ".join(terms))

# Call the solve function
solve()