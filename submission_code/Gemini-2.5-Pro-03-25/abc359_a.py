# YOUR CODE HERE
import sys

def solve():
    # Read the integer N from the first line
    n = int(sys.stdin.readline())

    # Initialize a counter for the name "Takahashi"
    takahashi_count = 0

    # Loop N times to read each string
    for _ in range(n):
        # Read the i-th string S_i
        # Use strip() to remove potential leading/trailing whitespace, including the newline character
        s_i = sys.stdin.readline().strip()

        # Check if the string is equal to "Takahashi"
        if s_i == "Takahashi":
            # If it is, increment the counter
            takahashi_count += 1

    # After reading all N strings, print the final count
    print(takahashi_count)

# Call the solve function to execute the logic
solve()
# YOUR CODE HERE