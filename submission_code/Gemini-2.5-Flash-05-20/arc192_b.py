import sys

def solve():
    N = int(sys.stdin.readline())
    # The A values are not needed for the logic, as explained above.
    # We only care about N's parity.
    # A = list(map(int, sys.stdin.readline().split()))

    if N % 2 == 1:
        print("Fennec")
    else:
        print("Snuke")

# Call the solve function to run the program
solve()