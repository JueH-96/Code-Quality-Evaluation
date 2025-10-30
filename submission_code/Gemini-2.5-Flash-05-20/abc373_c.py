import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read sequence A and find its maximum element
    # Using sys.stdin.readline() for potentially faster input
    A = list(map(int, sys.stdin.readline().split()))
    max_A = max(A)

    # Read sequence B and find its maximum element
    B = list(map(int, sys.stdin.readline().split()))
    max_B = max(B)

    # The maximum sum is the sum of the maximum elements from A and B
    result = max_A + max_B

    # Print the result
    print(result)

# Call the solve function to run the program
if __name__ == '__main__':
    solve()