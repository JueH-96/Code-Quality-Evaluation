# YOUR CODE HERE
import sys

def solve():
    # Read A and B from standard input
    A, B = map(int, sys.stdin.readline().split())

    # Initialize a set of all possible culprits
    suspects = {1, 2, 3}

    # Ringo's memory: Person A is not the culprit
    # Using discard() is safe because it won't raise an error if A is not in the set
    suspects.discard(A)

    # Snuke's memory: Person B is not the culprit
    suspects.discard(B)

    # Check the number of remaining possible culprits
    if len(suspects) == 1:
        # If only one suspect remains, they are the unique culprit
        # Convert set to list and get the first (and only) element
        print(list(suspects)[0])
    else:
        # If more than one suspect remains, the culprit cannot be uniquely identified
        print("-1")

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()