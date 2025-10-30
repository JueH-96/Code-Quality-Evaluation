import sys

def solve():
    """
    Reads a string from stdin and prints "Yes" if it ends with "san",
    otherwise prints "No".
    """
    # Read the input string from standard input
    s = sys.stdin.readline().strip()

    # Check if the string ends with "san"
    if s.endswith("san"):
        # If it does, print "Yes"
        print("Yes")
    else:
        # Otherwise, print "No"
        print("No")

# Call the solve function to run the program
solve()