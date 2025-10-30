def solve():
    import sys
    data = sys.stdin.read().strip().split()
    X, Y = map(int, data)

    if Y > X:
        # moving up
        if (Y - X) <= 2:
            print("Yes")
        else:
            print("No")
    else:
        # moving down
        if (X - Y) <= 3:
            print("Yes")
        else:
            print("No")

# Call the solve() function
solve()