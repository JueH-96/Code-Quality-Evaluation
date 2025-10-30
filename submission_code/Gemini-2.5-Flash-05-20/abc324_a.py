import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read the list of integers
    # Split the line by spaces and convert each part to an integer
    A = list(map(int, sys.stdin.readline().split()))

    # To check if all values are equal, we can convert the list to a set.
    # If all values are equal, the set will contain only one unique element.
    if len(set(A)) == 1:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the program
solve()