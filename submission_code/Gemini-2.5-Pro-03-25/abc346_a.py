import sys

def solve():
    # Read the integer N
    n = int(sys.stdin.readline())

    # Read the list of N integers A
    a = list(map(int, sys.stdin.readline().split()))

    # Calculate the list B using a list comprehension
    # B_i = A_i * A_{i+1} for 1 <= i <= N-1
    # In 0-based indexing, this corresponds to a[i] * a[i+1] for 0 <= i <= N-2
    b = [a[i] * a[i+1] for i in range(n - 1)]

    # Print the elements of B separated by spaces
    # The * operator unpacks the list b into individual arguments for print,
    # which prints them separated by spaces by default.
    print(*b)

# Call the solve function to run the logic
solve()