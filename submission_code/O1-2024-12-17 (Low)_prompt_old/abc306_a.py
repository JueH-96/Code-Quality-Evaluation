def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # Build the result by doubling each character in S
    result = "".join(char * 2 for char in S)
    
    print(result)

# Call the solve() function
solve()