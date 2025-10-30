def solve():
    import sys

    X = sys.stdin.readline().strip()
    # Convert to float to remove trailing zeros properly
    value = float(X)
    # Convert back to string, then remove trailing zeros and possibly trailing dot
    out_str = str(value).rstrip('0').rstrip('.')
    print(out_str)

# Call solve() function
solve()