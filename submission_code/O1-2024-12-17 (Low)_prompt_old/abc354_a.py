def solve():
    import sys
    H = int(sys.stdin.readline().strip())
    
    day = 0
    height = 0
    
    # On the morning of day 'i', the plant's height is (2^i - 1).
    # We need the first day 'i' such that (2^i - 1) > H.
    # Use a simple loop until the condition is satisfied.
    while height <= H:
        day += 1
        height = (1 << day) - 1  # Equivalent to 2^day - 1

    print(day)

# Call solve() to execute
solve()