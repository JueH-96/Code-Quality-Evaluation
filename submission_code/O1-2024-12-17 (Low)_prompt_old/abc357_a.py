def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    H = list(map(int, data[2:]))

    remaining = M
    count = 0

    for hands in H:
        if hands <= remaining:
            count += 1
            remaining -= hands
        else:
            # This alien uses whatever is left, failing to disinfect all hands
            break

    print(count)

# Call the solve() function
solve()