def solve():
    # Read input from stdin
    x, y = map(int, input().split())

    # Calculate the absolute difference between the two floors
    diff = abs(y - x)

    # Check if the difference is within the range where Takahashi uses the stairs
    if (y > x and diff <= 2) or (y < x and diff <= 3):
        print("Yes")
    else:
        print("No")

# Call the solve function
solve()